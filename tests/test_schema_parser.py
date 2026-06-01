"""Offline tests for RADIOLOGY_JSON_SCHEMA and parse_to_analysis().

No Gemini call, no API key required.
Run: pytest tests/test_schema_parser.py
"""
import pytest
from radiology_pipeline import parse_to_analysis


def _valid_raw(**overrides):
    base = {
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
    base.update(overrides)
    return base


def test_happy_path():
    analysis = parse_to_analysis(_valid_raw())
    assert analysis.label == "Chest X-ray — Normal study"
    assert analysis.confidence == "High"
    assert analysis.metadata["is_medical_image"] is True
    assert analysis.metadata["modality"] == "Chest X-ray"
    assert analysis.metadata["view"] == "PA"
    assert analysis.metadata["severity_list"] == ["normal"]


def test_empty_per_structure_findings():
    analysis = parse_to_analysis(_valid_raw(per_structure_findings=[]))
    assert analysis.metadata["severity_list"] == []
    assert analysis.metadata["per_structure"] == []


def test_confidence_normalization_lowercase():
    analysis = parse_to_analysis(_valid_raw(confidence_level="low"))
    assert analysis.confidence == "Low"


def test_confidence_normalization_uppercase():
    analysis = parse_to_analysis(_valid_raw(confidence_level="HIGH"))
    assert analysis.confidence == "High"


def test_missing_required_key_raises():
    raw = _valid_raw()
    del raw["impression"]
    with pytest.raises(KeyError):
        parse_to_analysis(raw)


def test_is_medical_false_preserved():
    analysis = parse_to_analysis(_valid_raw(is_medical_image=False))
    assert analysis.metadata["is_medical_image"] is False


def test_multiple_per_structure_severity_list():
    findings = [
        {"structure": "Lungs", "observation": "Clear", "severity": "normal"},
        {"structure": "Heart", "observation": "Enlarged", "severity": "moderate"},
        {"structure": "Aorta", "observation": "Ectatic", "severity": "critical"},
    ]
    analysis = parse_to_analysis(_valid_raw(per_structure_findings=findings))
    assert analysis.metadata["severity_list"] == ["normal", "moderate", "critical"]
