"""One-time export of the TorchXRayVision chest-X-ray classifier to ONNX.

Run this **once on a dev machine** (it needs PyTorch + torchxrayvision, which are
*not* deployed). It downloads the pretrained DenseNet weights (~120 MB), wraps the
model so its output is plain pathology probabilities in a fixed order, and writes
``models/chexnet.onnx``. Production then needs only onnxruntime — no PyTorch.

    pip install -r requirements-export.txt
    python tools/export_onnx.py

Commit the resulting models/chexnet.onnx (or attach it to a release and download
it at deploy time) so Streamlit Cloud never installs torch.
"""
import os
import sys

import torch
import torchxrayvision as xrv

# Must match local_backend.PATHOLOGIES exactly — index i of the ONNX output maps
# to PATHOLOGIES[i] at inference time, so the order is load-bearing.
EXPECTED_PATHOLOGIES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia", "Lung Lesion", "Fracture",
    "Lung Opacity", "Enlarged Cardiomediastinum",
]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
OUT_PATH = os.path.join(OUT_DIR, "chexnet.onnx")


def main() -> int:
    print("Loading densenet121-res224-all (downloads weights on first run)…")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()

    pathologies = list(model.pathologies)
    if pathologies != EXPECTED_PATHOLOGIES:
        print(
            "ERROR: model pathology order does not match local_backend.PATHOLOGIES.\n"
            f"  model:    {pathologies}\n"
            f"  expected: {EXPECTED_PATHOLOGIES}\n"
            "Update PATHOLOGIES in both files to match before exporting.",
            file=sys.stderr,
        )
        return 1

    # DenseNet.forward calls utils.warn_normalization(x), which does a
    # data-dependent `.item()` on the input to sanity-check the pixel range. That
    # CPU sync injects an unbacked SymInt into the trace and breaks ONNX export
    # ("Sym(Eq(u0, 1))" / "Unable to find user code corresponding to {u0}"). It
    # has no effect on the model's output, so we replace it with a no-op for the
    # export. We patch the `utils` module attribute that forward looks up at call
    # time (the traceback shows `utils.warn_normalization(x)`), so this takes
    # effect regardless of how the function was imported.
    xrv.utils.warn_normalization = lambda *args, **kwargs: None

    # The "all" model carries op_threshs and, when set, forward applies a final
    # op_norm() calibration step. op_norm indexes over only the pathologies with
    # a defined (non-NaN) threshold — a data-dependent gather/reshape that ONNX
    # can't trace statically (it fails at runtime with a Reshape mismatch, e.g.
    # input {15} vs the 18 outputs). Disabling it makes forward return the raw
    # per-pathology logits; local_backend.predict_probabilities applies the
    # sigmoid in numpy, yielding standard probabilities in [0, 1].
    model.op_threshs = None

    os.makedirs(OUT_DIR, exist_ok=True)
    dummy = torch.randn(1, 1, 224, 224)  # (batch, channel, H, W), xrv input shape

    export_kwargs = dict(
        input_names=["image"],
        output_names=["probabilities"],
        # Allow variable batch size; H/W are fixed at 224 by the model.
        dynamic_axes={"image": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=13,
        do_constant_folding=True,
    )

    # Recent PyTorch defaults torch.onnx.export to the dynamo/torch.export path,
    # which is stricter about data-dependent ops. This CNN exports cleanly with
    # the stable TorchScript tracer, so force it. Older torch lacks the `dynamo`
    # kwarg and already uses TorchScript, so fall back without it.
    try:
        torch.onnx.export(model, dummy, OUT_PATH, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, OUT_PATH, **export_kwargs)

    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"Wrote {OUT_PATH} ({size_mb:.0f} MB) with {len(pathologies)} outputs.")

    # Best-effort parity check: confirm the exported graph reproduces PyTorch's
    # output (guards against a silently corrupt / partially-traced graph). Skipped
    # if onnxruntime isn't installed in the export env.
    try:
        import numpy as np
        import onnxruntime as ort

        with torch.no_grad():
            torch_out = model(dummy).numpy()
        sess = ort.InferenceSession(OUT_PATH, providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {"image": dummy.numpy()})[0]
        max_diff = float(np.abs(torch_out - onnx_out).max())
        if onnx_out.shape[-1] != len(EXPECTED_PATHOLOGIES) or max_diff > 1e-3:
            print(
                f"WARNING: ONNX/PyTorch parity check failed "
                f"(shape={onnx_out.shape}, max_diff={max_diff:.2e}).",
                file=sys.stderr,
            )
            return 1
        print(f"Parity check OK (max abs diff {max_diff:.2e}).")
    except ImportError:
        print("Parity check skipped (onnxruntime not installed in export env).")

    print("Done. Production needs only: pip install onnxruntime numpy pillow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
