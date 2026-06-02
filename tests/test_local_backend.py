"""Offline tests for the local CPU chest-X-ray backend.

These never touch onnxruntime or the .onnx file: ONNX inference is monkeypatched,
and the pure mapping/gate logic is exercised directly. The key guarantee tested
is that build_report() output is consumable by the real schema parser and that
the existing vlm-guard rules fire correctly off it.

Run: pytest tests/test_local_backend.py
"""
import json

import numpy as np
import pytest
from PIL import Image

import local_backend as lb
from radiology_pipeline import RADIOLOGY_JSON_SCHEMA, build_local_pipeline, parse_to_analysis

_SEVERITY_ENUM = set(
    RADIOLOGY_JSON_SCHEMA["properties"]["per_structure_findings"]["items"][
        "properties"
    ]["severity"]["enum"]
)
_CONFIDENCE_ENUM = set(
    RADIOLOGY_JSON_SCHEMA["properties"]["confidence_level"]["enum"]
)


def _assert_schema_shaped(report: dict):
    """Cheap structural validation against RADIOLOGY_JSON_SCHEMA's contract."""
    for key in RADIOLOGY_JSON_SCHEMA["required"]:
        assert key in report, f"missing required field: {key}"
    assert report["confidence_level"] in _CONFIDENCE_ENUM
    for f in report["per_structure_findings"]:
        assert {"structure", "observation", "severity"} <= f.keys()
        assert f["severity"] in _SEVERITY_ENUM
    # The real consumer must accept it without raising.
    parse_to_analysis(report)


# ── is_medical_image gate ─────────────────────────────────────────────────────


def test_gate_rejects_colour_image():
    red = Image.new("RGB", (64, 64), color=(255, 0, 0))
    assert lb.looks_like_xray(red) is False


def test_gate_rejects_flat_image():
    flat = Image.new("RGB", (64, 64), color=(128, 128, 128))
    assert lb.looks_like_xray(flat) is False


def test_gate_accepts_greyscale_with_contrast():
    rng = np.random.default_rng(0)
    noise = (rng.random((64, 64)) * 255).astype(np.uint8)
    img = Image.fromarray(noise, mode="L").convert("RGB")
    assert lb.looks_like_xray(img) is True


def _synthetic_ct_slice():
    """Black frame with a bright circular field of view — axial CT signature."""
    yy, xx = np.mgrid[0:256, 0:256]
    disk = ((xx - 128) ** 2 + (yy - 128) ** 2) < 110 ** 2
    arr = np.zeros((256, 256), np.uint8)
    arr[disk] = 160
    return Image.fromarray(arr, mode="L").convert("RGB")


def test_ct_slice_detected():
    ct = _synthetic_ct_slice()
    assert lb.looks_like_xray(ct) is True       # greyscale w/ contrast — passes the basic gate
    assert lb.looks_like_ct_slice(ct) is True   # ...but the CT detector catches it


def test_full_frame_noise_not_flagged_as_ct():
    rng = np.random.default_rng(2)
    noise = (rng.random((96, 96)) * 255).astype(np.uint8)
    img = Image.fromarray(noise, mode="L").convert("RGB")
    assert lb.looks_like_ct_slice(img) is False  # bright corners → not a circular FOV


def test_ct_rejected_with_modality_message(monkeypatch):
    monkeypatch.setattr(lb, "looks_like_xray", lambda img: True)
    monkeypatch.setattr(lb, "looks_like_ct_slice", lambda img: True)
    report = json.loads(lb.local_model_fn(Image.new("RGB", (64, 64)), "x"))
    assert report["is_medical_image"] is False
    assert "CT" in report["modality"]
    assert "Gemini" in report["recommendation"]
    parse_to_analysis(report)  # must stay schema-valid


# ── severity bucketing ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "prob,expected",
    [(0.55, "mild"), (0.60, "moderate"), (0.75, "severe"), (0.90, "critical"),
     (0.99, "critical")],
)
def test_severity_buckets(prob, expected):
    assert lb._bucket_severity(prob) == expected


# ── build_report ──────────────────────────────────────────────────────────────


def test_report_normal_when_all_below_threshold():
    probs = {name: 0.1 for name in lb.PATHOLOGIES}
    report = lb.build_report(probs, is_medical=True)
    _assert_schema_shaped(report)
    assert report["is_medical_image"] is True
    assert report["per_structure_findings"] == []
    assert "No acute" in report["impression"]
    assert report["confidence_level"] == "High"  # decisively negative read


def test_report_flags_strong_finding():
    probs = {name: 0.1 for name in lb.PATHOLOGIES}
    probs["Cardiomegaly"] = 0.92
    report = lb.build_report(probs, is_medical=True)
    _assert_schema_shaped(report)
    findings = report["per_structure_findings"]
    assert len(findings) == 1
    assert findings[0]["structure"] == "Heart"
    assert findings[0]["severity"] == "critical"
    assert report["confidence_level"] == "High"
    assert "Urgent" in report["recommendation"]


def test_report_headlines_primary_and_folds_associated():
    probs = {name: 0.1 for name in lb.PATHOLOGIES}
    probs.update({"Lung Opacity": 0.91, "Consolidation": 0.82, "Effusion": 0.62})
    report = lb.build_report(probs, is_medical=True)
    _assert_schema_shaped(report)
    # Strongest finding headlines the impression; the rest are summarised as associated.
    assert report["impression"].startswith("Lung Opacity (critical)")
    assert "2 associated finding(s)" in report["impression"]
    assert report["key_findings"].startswith("Primary: Lung Opacity (91%)")
    assert "Associated:" in report["key_findings"]
    # Full per-structure list is retained (primary first) so guardrail rules still see everything.
    assert len(report["per_structure_findings"]) == 3
    assert report["per_structure_findings"][0]["observation"].startswith("Lung Opacity")


def test_report_findings_sorted_by_probability():
    probs = {name: 0.1 for name in lb.PATHOLOGIES}
    probs["Effusion"] = 0.7
    probs["Cardiomegaly"] = 0.95
    report = lb.build_report(probs, is_medical=True)
    severities = [f["observation"] for f in report["per_structure_findings"]]
    assert severities[0].startswith("Cardiomegaly")  # highest prob first


def test_report_non_medical():
    report = lb.build_report({}, is_medical=False)
    _assert_schema_shaped(report)
    assert report["is_medical_image"] is False
    assert report["per_structure_findings"] == []


# ── full pipeline (ONNX monkeypatched) ────────────────────────────────────────


def _grey_image():
    rng = np.random.default_rng(1)
    noise = (rng.random((128, 128)) * 255).astype(np.uint8)
    return Image.fromarray(noise, mode="L").convert("RGB")


def test_pipeline_runs_with_local_backend(monkeypatch):
    probs = {name: 0.1 for name in lb.PATHOLOGIES}
    probs["Cardiomegaly"] = 0.92
    monkeypatch.setattr(lb, "looks_like_xray", lambda img: True)
    monkeypatch.setattr(lb, "predict_probabilities", lambda img: probs)

    pipeline = build_local_pipeline()
    result = pipeline.run(_grey_image(), "Analyze.", context={"scan_type": "radiology"})

    assert result.analysis.metadata["is_medical_image"] is True
    assert "critical" in result.analysis.metadata["severity_list"]
    assert isinstance(result.audit.summary(), list)


def test_pipeline_blocks_non_xray(monkeypatch):
    monkeypatch.setattr(lb, "looks_like_xray", lambda img: False)

    pipeline = build_local_pipeline()
    result = pipeline.run(
        Image.new("RGB", (64, 64), color=(255, 0, 0)),
        "Analyze.",
        context={"scan_type": "radiology"},
    )

    actions = [e["action"] for e in result.audit.summary()]
    assert "block" in actions
    assert result.analysis.metadata["is_medical_image"] is False
