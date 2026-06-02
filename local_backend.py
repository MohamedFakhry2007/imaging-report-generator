"""Local, CPU-only chest X-ray backend — a drop-in replacement for the Gemini
``model_fn`` used by :func:`radiology_pipeline.build_local_pipeline`.

Why this exists
---------------
``gemini-2.0-flash`` does two jobs at once: it *sees* the scan and *emits*
schema JSON. Neither half fits a 1-vCPU / ~1 GB Streamlit Cloud free tier or a
weak local CPU. So we split them:

* **Vision** is a ~30M-param TorchXRayVision DenseNet (``densenet121-res224-all``)
  exported to ONNX (see ``tools/export_onnx.py``). It runs in well under a second
  on CPU via :mod:`onnxruntime` — no PyTorch needed at run time, which keeps the
  deployed RSS comfortably under 1 GB.
* **Structured output** is plain Python: the classifier's 18 pathology
  probabilities are templated into ``RADIOLOGY_JSON_SCHEMA`` JSON. No generative
  model means valid JSON by construction and zero hallucinated prose.

Scope: chest radiographs only. Anything else is rejected by :func:`looks_like_xray`
so the pipeline's ``NonMedicalImageRule`` blocks it rather than inventing findings.

The pure logic (:func:`looks_like_xray`, :func:`build_report`) is deliberately
separated from ONNX inference so it can be unit-tested offline without the model
file or onnxruntime installed.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

# ── Canonical model output order ──────────────────────────────────────────────
# TorchXRayVision densenet121-res224-all pathology order. tools/export_onnx.py
# asserts the loaded model matches this exactly, so index i of the ONNX output
# always corresponds to PATHOLOGIES[i].
PATHOLOGIES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia", "Lung Lesion", "Fracture",
    "Lung Opacity", "Enlarged Cardiomediastinum",
]

# Map each pathology to the anatomical structure reported in per_structure_findings.
_STRUCTURE = {
    "Cardiomegaly": "Heart",
    "Enlarged Cardiomediastinum": "Heart / mediastinum",
    "Effusion": "Pleura",
    "Pleural_Thickening": "Pleura",
    "Pneumothorax": "Pleura",
    "Atelectasis": "Lungs",
    "Consolidation": "Lungs",
    "Infiltration": "Lungs",
    "Edema": "Lungs",
    "Pneumonia": "Lungs",
    "Lung Opacity": "Lungs",
    "Lung Lesion": "Lungs",
    "Nodule": "Lungs",
    "Mass": "Lungs",
    "Emphysema": "Lung parenchyma",
    "Fibrosis": "Lung parenchyma",
    "Fracture": "Bony thorax",
    "Hernia": "Diaphragm",
}

# ── Tunables (overridable via environment) ────────────────────────────────────
# Probability above which a pathology is reported as a finding.
DETECTION_THRESHOLD = float(os.environ.get("CHEXNET_THRESHOLD", "0.5"))
# Heuristic gate thresholds (see looks_like_xray).
_MAX_SATURATION = float(os.environ.get("CHEXNET_MAX_SATURATION", "15"))
_MIN_CONTRAST = float(os.environ.get("CHEXNET_MIN_CONTRAST", "10"))

_ONNX_PATH = os.environ.get(
    "CHEXNET_ONNX_PATH",
    os.path.join(os.path.dirname(__file__), "models", "chexnet.onnx"),
)

# Lazily-initialised onnxruntime session (heavy import; kept out of module load).
_session = None


# ── is_medical_image gate ─────────────────────────────────────────────────────


def looks_like_xray(image: Image.Image) -> bool:
    """Cheap heuristic: does this image plausibly look like a chest radiograph?

    Radiographs are greyscale (near-zero colour saturation) and have real tonal
    range (non-trivial contrast). A colour photo fails the saturation test; a
    solid/flat image fails the contrast test. This is intentionally conservative
    for a CXR-only tool — it will not catch a greyscale *non-medical* photo, but
    it reliably rejects colour images and blank uploads. Limitations are
    documented; downstream the NonMedicalImageRule blocks anything that fails.
    """
    small = np.asarray(image.convert("RGB").resize((64, 64))).astype(np.float32)
    # Per-pixel (max-min) across channels ≈ saturation; ~0 for true greyscale.
    saturation = float((small.max(axis=2) - small.min(axis=2)).mean())
    contrast = float(small.mean(axis=2).std())
    return saturation < _MAX_SATURATION and contrast > _MIN_CONTRAST


def looks_like_ct_slice(image: Image.Image) -> bool:
    """Best-effort detector for an axial CT slice (which the CXR model can't read).

    Both X-rays and CT are greyscale, so looks_like_xray() can't separate them.
    But an axial CT reconstruction has a *circular field of view*: the corners sit
    outside the bore and are pure black, while the mid-edges and centre (the body
    cross-section) are bright. Chest radiographs fill the frame, so their corners
    are not uniformly black with a bright interior.

    This catches the common axial-CT case. It is NOT bulletproof — a CT exported
    to fill the frame, or a coronal/sagittal reformat, may slip through, and the
    Low-confidence guardrail is the backstop for those. Thresholds are tunable.
    """
    arr = np.asarray(image.convert("L").resize((96, 96)), dtype=np.float32)
    c = 14  # corner / edge patch size
    mid = slice(48 - c // 2, 48 + c // 2)

    corner_mean = float(np.mean([
        arr[:c, :c].mean(), arr[:c, -c:].mean(),
        arr[-c:, :c].mean(), arr[-c:, -c:].mean(),
    ]))
    edge_mean = float(np.mean([
        arr[:c, mid].mean(), arr[-c:, mid].mean(),
        arr[mid, :c].mean(), arr[mid, -c:].mean(),
    ]))
    center_mean = float(arr[mid, mid].mean())

    dark_corners = corner_mean < 25.0
    bright_interior = (center_mean > corner_mean + 40.0
                       and edge_mean > corner_mean + 25.0)
    return dark_corners and bright_interior


# ── Probability → schema mapping (pure, unit-testable) ────────────────────────


def _bucket_severity(prob: float) -> str:
    """Map a pathology probability to the schema's severity enum."""
    if prob >= 0.90:
        return "critical"
    if prob >= 0.75:
        return "severe"
    if prob >= 0.60:
        return "moderate"
    return "mild"  # in [threshold, 0.60)


def _confidence(probs: dict[str, float], threshold: float) -> str:
    """Derive an overall confidence_level from how decisive the probabilities are.

    Findings present → confidence tracks the strongest finding. No findings →
    confidence reflects how cleanly everything sits below threshold (values that
    hover just under it mean an uncertain "normal").
    """
    if not probs:
        return "Low"
    top = max(probs.values())
    flagged = [p for p in probs.values() if p >= threshold]
    if flagged:
        strongest = max(flagged)
        if strongest >= 0.85:
            return "High"
        if strongest >= 0.65:
            return "Medium"
        return "Low"
    # Clean negative read: penalise borderline probabilities near the threshold.
    if top < threshold - 0.15:
        return "High"
    if top < threshold - 0.05:
        return "Medium"
    return "Low"


def build_report(
    probs: dict[str, float],
    is_medical: bool,
    threshold: float = DETECTION_THRESHOLD,
    *,
    unsupported_modality: bool = False,
) -> dict:
    """Template classifier probabilities into a RADIOLOGY_JSON_SCHEMA-shaped dict.

    ``probs`` maps pathology name → probability in [0, 1]. ``is_medical`` is the
    output of the :func:`looks_like_xray` gate. ``unsupported_modality`` marks a
    rejection where the input *is* medical but not a chest radiograph (e.g. a CT
    slice) so the message can say so rather than "not a medical image". The
    returned dict contains every field ``radiology_pipeline.parse_to_analysis``
    requires.
    """
    if not is_medical:
        if unsupported_modality:
            modality = "CT / cross-sectional (unsupported)"
            impression = "Input appears to be a CT or cross-sectional scan, not a chest radiograph."
            recommendation = (
                "The local backend supports chest X-rays only. "
                "Use the Gemini backend for CT/MRI."
            )
        else:
            modality = "Unknown"
            impression = "Input does not appear to be a chest radiograph."
            recommendation = "Upload a chest X-ray (greyscale radiograph) for analysis."
        return {
            "modality": modality,
            "view": "Unknown",
            "is_medical_image": False,
            "impression": impression,
            "confidence_level": "Low",
            "key_findings": "No analysis performed — unsupported or non-medical image.",
            "per_structure_findings": [],
            "recommendation": recommendation,
        }

    findings = sorted(
        ((name, p) for name, p in probs.items() if p >= threshold),
        key=lambda kv: kv[1],
        reverse=True,
    )

    per_structure = [
        {
            "structure": _STRUCTURE.get(name, "Chest"),
            "observation": f"{name.replace('_', ' ')} (probability {p:.0%})",
            "severity": _bucket_severity(p),
        }
        for name, p in findings
    ]

    confidence = _confidence(probs, threshold)

    if findings:
        # findings is sorted by probability, so findings[0] is the dominant read.
        # The CheXNet labels are correlated (one opacity often fires several), so
        # we headline the strongest and treat the rest as associated/differential.
        (primary_name, primary_p), associated = findings[0], findings[1:]
        primary_label = primary_name.replace("_", " ")
        primary_sev = _bucket_severity(primary_p)

        if associated:
            impression = (
                f"{primary_label} ({primary_sev}); "
                f"{len(associated)} associated finding(s)."
            )
            key_findings = (
                f"Primary: {primary_label} ({primary_p:.0%}). Associated: "
                + ", ".join(f"{n.replace('_', ' ')} ({p:.0%})" for n, p in associated)
                + "."
            )
        else:
            impression = f"{primary_label} ({primary_sev})."
            key_findings = f"Primary: {primary_label} ({primary_p:.0%})."

        severities = {f["severity"] for f in per_structure}
        if severities & {"severe", "critical"}:
            recommendation = "Urgent radiologist review recommended."
        else:
            recommendation = "Correlate clinically; radiologist review advised."
    else:
        impression = "No acute cardiopulmonary abnormality detected."
        key_findings = "No pathology above the detection threshold."
        recommendation = "No acute findings; routine follow-up as clinically indicated."

    return {
        "modality": "Chest X-ray",
        "view": "Unknown",  # the classifier does not infer projection
        "is_medical_image": True,
        "impression": impression,
        "confidence_level": confidence,
        "key_findings": key_findings,
        "per_structure_findings": per_structure,
        "recommendation": recommendation,
    }


# ── ONNX inference ────────────────────────────────────────────────────────────


def ensure_model_available() -> None:
    """Raise FileNotFoundError (with export instructions) if the ONNX model is
    missing. Lets the UI report the problem up front, before any image is run.
    """
    if not os.path.exists(_ONNX_PATH):
        raise FileNotFoundError(
            f"Local CXR model not found at {_ONNX_PATH}. Generate it once with:\n"
            "    pip install -r requirements-export.txt\n"
            "    python tools/export_onnx.py"
        )


def _load_session():
    """Lazily create the onnxruntime session (single CPU thread is plenty)."""
    global _session
    if _session is None:
        if not os.path.exists(_ONNX_PATH):
            raise FileNotFoundError(
                f"Local CXR model not found at {_ONNX_PATH}. Generate it once with:\n"
                "    pip install -r requirements-export.txt\n"
                "    python tools/export_onnx.py"
            )
        import onnxruntime as ort  # heavy; imported only when actually inferring

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # weak CPUs: avoid oversubscription
        _session = ort.InferenceSession(
            _ONNX_PATH, sess_options=opts, providers=["CPUExecutionProvider"]
        )
    return _session


def _preprocess(image: Image.Image) -> np.ndarray:
    """Replicate TorchXRayVision preprocessing: greyscale, centre-cropped to a
    square, resized to 224×224, normalised to the [-1024, 1024] range xrv uses.
    Returns a (1, 1, 224, 224) float32 array.
    """
    gray = image.convert("L")
    w, h = gray.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    gray = gray.crop((left, top, left + side, top + side)).resize((224, 224))

    arr = np.asarray(gray, dtype=np.float32)          # [0, 255]
    arr = (2.0 * (arr / 255.0) - 1.0) * 1024.0         # xrv normalize → [-1024, 1024]
    return arr[None, None, :, :]                       # (1, 1, 224, 224)


def predict_probabilities(image: Image.Image) -> dict[str, float]:
    """Run the ONNX classifier and return {pathology: probability}."""
    session = _load_session()
    inp = _preprocess(image)
    input_name = session.get_inputs()[0].name
    logits = np.asarray(session.run(None, {input_name: inp})[0]).ravel()

    # The exported model returns raw per-pathology logits (op_threshs is disabled
    # at export time — see tools/export_onnx.py), so apply the sigmoid here to get
    # probabilities. The guard also covers a model that already applied sigmoid:
    # if everything is already in [0, 1], leave it untouched.
    if logits.min() < 0.0 or logits.max() > 1.0:
        logits = 1.0 / (1.0 + np.exp(-logits))

    return {name: float(p) for name, p in zip(PATHOLOGIES, logits)}


# ── Pipeline entry point ──────────────────────────────────────────────────────


def local_model_fn(image: Image.Image, prompt: str) -> str:
    """model_fn for VLMGuardPipeline: (image, prompt) → schema JSON string.

    ``prompt`` is ignored — the classifier needs no instructions — but kept in
    the signature to match the Gemini backend so the two are interchangeable.
    """
    if not looks_like_xray(image):
        return json.dumps(build_report({}, is_medical=False))
    if looks_like_ct_slice(image):
        return json.dumps(build_report({}, is_medical=False, unsupported_modality=True))
    probs = predict_probabilities(image)
    return json.dumps(build_report(probs, is_medical=True))
