"""Offline tests for parse_raw() — the string→(Analysis, str) adapter that
matches VLMGuardPipeline's parser_fn contract.

No Gemini call, no API key, no ONNX model required.
Run: pytest tests/test_parse_raw.py
"""
import json

import pytest

from radiology_pipeline import Analysis, parse_raw

_VALID = {
    "modality": "Chest X-ray",
    "view": "PA",
    "is_medical_image": True,
    "impression": "Normal study",
    "confidence_level": "High",
    "key_findings": "Lungs clear bilaterally.",
    "per_structure_findings": [
        {"structure": "Lungs", "observation": "Clear", "severity": "normal"},
    ],
    "recommendation": "No follow-up required.",
}


def test_returns_analysis_and_raw_tuple():
    raw = json.dumps(_VALID)
    result = parse_raw(raw)
    assert isinstance(result, tuple) and len(result) == 2
    analysis, raw_output = result
    assert isinstance(analysis, Analysis)
    assert raw_output == raw  # the original string is passed through unchanged


def test_unpacks_like_the_pipeline_does():
    # Mirrors vlm_guard.core.pipeline.run: `analysis, raw_output = parser_fn(raw)`
    analysis, raw_output = parse_raw(json.dumps(_VALID))
    assert analysis.label == "Chest X-ray — Normal study"
    assert analysis.metadata["severity_list"] == ["normal"]


def test_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_raw("not json")
