"""Offline tests for vlm-guard rules and GuardrailEngine.

No Gemini call, no API key required.
Run: pytest tests/test_rules.py
"""
from unittest.mock import MagicMock

import pytest
from radiology_pipeline import (
    LowConfidenceRule,
    NonMedicalImageRule,
    SeverityConsistencyRule,
    engine,
)


def _make_analysis(
    label="Chest X-ray — Normal study",
    confidence="High",
    recommendation="No follow-up required.",
    is_medical=True,
    severity_list=None,
):
    analysis = MagicMock()
    analysis.label = label
    analysis.confidence = confidence
    analysis.recommendation = recommendation
    analysis.metadata = {
        "is_medical_image": is_medical,
        "severity_list": severity_list or [],
    }
    return analysis


# ── NonMedicalImageRule ───────────────────────────────────────────────────────

class TestNonMedicalImageRule:
    rule = NonMedicalImageRule()

    def test_fires_when_not_medical(self):
        analysis = _make_analysis(is_medical=False)
        assert self.rule.condition(analysis, {}) is True
        modified, result = self.rule.action(analysis, {})
        assert "BLOCKED" in modified.label
        assert modified.confidence == "Low"
        assert result.action_type == "block"
        assert result.action_taken is True

    def test_no_fire_when_medical(self):
        analysis = _make_analysis(is_medical=True)
        assert self.rule.condition(analysis, {}) is False


# ── LowConfidenceRule ─────────────────────────────────────────────────────────

class TestLowConfidenceRule:
    rule = LowConfidenceRule()

    def test_fires_when_low_confidence(self):
        analysis = _make_analysis(confidence="Low", is_medical=True)
        assert self.rule.condition(analysis, {}) is True
        modified, result = self.rule.action(analysis, {})
        assert "⚠️" in modified.recommendation
        assert "senior radiologist review" in modified.recommendation
        assert result.action_type == "flag"
        assert result.action_taken is True

    def test_no_fire_when_high_confidence(self):
        analysis = _make_analysis(confidence="High", is_medical=True)
        assert self.rule.condition(analysis, {}) is False

    def test_no_fire_when_medium_confidence(self):
        analysis = _make_analysis(confidence="Medium", is_medical=True)
        assert self.rule.condition(analysis, {}) is False

    def test_no_fire_when_non_medical_blocked(self):
        # NonMedicalImageRule fires first and sets confidence="Low".
        # LowConfidenceRule must not also fire — the guard in condition() prevents it.
        analysis = _make_analysis(confidence="Low", is_medical=False)
        assert self.rule.condition(analysis, {}) is False


# ── SeverityConsistencyRule ───────────────────────────────────────────────────

class TestSeverityConsistencyRule:
    rule = SeverityConsistencyRule()

    def test_fires_with_critical_and_normal_label(self):
        analysis = _make_analysis(
            label="Chest X-ray — Normal study",
            severity_list=["critical"],
        )
        assert self.rule.condition(analysis, {}) is True
        modified, result = self.rule.action(analysis, {})
        assert modified.confidence == "Low"
        assert "manual review" in modified.recommendation
        assert result.action_type == "correct"
        assert result.action_taken is True

    def test_no_fire_without_critical_severity(self):
        analysis = _make_analysis(
            label="Chest X-ray — Normal study",
            severity_list=["normal", "mild"],
        )
        assert self.rule.condition(analysis, {}) is False

    def test_no_fire_when_label_does_not_say_normal(self):
        analysis = _make_analysis(
            label="Chest X-ray — Pneumonia suspected",
            severity_list=["critical"],
        )
        assert self.rule.condition(analysis, {}) is False

    def test_no_fire_with_empty_severity_list(self):
        analysis = _make_analysis(
            label="Chest X-ray — Normal study",
            severity_list=[],
        )
        assert self.rule.condition(analysis, {}) is False


# ── Full engine (apply_with_audit) ────────────────────────────────────────────

class TestGuardrailEngine:
    def test_clean_analysis_zero_actions(self):
        """Clean scan: none of the three rules fire."""
        from radiology_pipeline import parse_to_analysis
        raw = {
            "modality": "Chest X-ray",
            "view": "PA",
            "is_medical_image": True,
            "impression": "Clear lungs",
            "confidence_level": "High",
            "key_findings": "No abnormalities.",
            "per_structure_findings": [
                {"structure": "Lungs", "observation": "Clear", "severity": "normal"},
            ],
            "recommendation": "Routine follow-up.",
        }
        analysis = parse_to_analysis(raw)
        _, audit = engine.apply_with_audit(analysis, {})
        fired = [e for e in audit.summary() if e.get("modified")]
        assert len(fired) == 0
