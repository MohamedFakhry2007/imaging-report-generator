"""Integration tests for VLMGuardPipeline with a real Gemini API call.

Requires GOOGLE_API_KEY environment variable.
Run:  GOOGLE_API_KEY=<key> pytest -m integration
Skip: pytest -m "not integration"
"""
import os
from pathlib import Path

import pytest
from PIL import Image

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

_SYSTEM_INSTRUCTION = (
    "If this is not a medical image, set is_medical_image to false "
    "and leave per_structure_findings as an empty list."
)


def _build_test_pipeline():
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=_SYSTEM_INSTRUCTION,
    )
    from radiology_pipeline import build_pipeline
    return build_pipeline(model)


@pytest.mark.integration
@pytest.mark.skipif(not GOOGLE_API_KEY, reason="requires GOOGLE_API_KEY env var")
def test_real_scan_returns_valid_analysis():
    pipeline = _build_test_pipeline()
    test_image_path = Path(__file__).parent.parent / "backend" / "test_image.jpg"
    if not test_image_path.exists():
        pytest.skip("backend/test_image.jpg not found")

    image = Image.open(test_image_path).convert("RGB")
    result = pipeline.run(image, "Analyze this medical image.", context={"scan_type": "radiology"})

    assert result.analysis.label, "label must be non-empty"
    required_keys = {"modality", "view", "is_medical_image", "per_structure", "severity_list"}
    assert required_keys.issubset(result.analysis.metadata.keys())
    assert isinstance(result.audit.summary(), list)


@pytest.mark.integration
@pytest.mark.skipif(not GOOGLE_API_KEY, reason="requires GOOGLE_API_KEY env var")
def test_non_medical_image_sets_flag():
    """Upload a clearly non-medical image; pipeline should set is_medical_image=False."""
    from io import BytesIO
    pipeline = _build_test_pipeline()

    # Create a 64x64 solid-colour PNG — clearly not a medical scan.
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    result = pipeline.run(img, "Analyze this medical image.", context={"scan_type": "radiology"})

    assert result.analysis.metadata["is_medical_image"] is False
