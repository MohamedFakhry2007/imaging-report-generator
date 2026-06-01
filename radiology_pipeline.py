from PIL import Image
from vlm_guard import Analysis, BaseRule, GuardrailEngine, RuleResult, VLMGuardPipeline
from vlm_guard.image.enhance import EnhancementStrategy, ImageEnhancer

# ── Schema ──────────────────────────────────────────────────────────────────

RADIOLOGY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "modality":            {"type": "string"},
        "view":                {"type": "string"},
        "is_medical_image":    {"type": "boolean"},
        "impression":          {"type": "string"},
        "confidence_level":    {"type": "string", "enum": ["High", "Medium", "Low"]},
        "key_findings":        {"type": "string"},
        "per_structure_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "structure":   {"type": "string"},
                    "observation": {"type": "string"},
                    "severity":    {
                        "type": "string",
                        "enum": ["normal", "mild", "moderate", "severe", "critical"],
                    },
                },
                "required": ["structure", "observation", "severity"],
            },
        },
        "recommendation": {"type": "string"},
    },
    "required": [
        "modality", "view", "is_medical_image", "impression",
        "confidence_level", "key_findings", "per_structure_findings", "recommendation",
    ],
}

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",  "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",  "threshold": "BLOCK_NONE"},
]

# ── Parser ───────────────────────────────────────────────────────────────────


def parse_to_analysis(raw: dict) -> Analysis:
    """Map a Gemini JSON response dict to a vlm-guard Analysis object.

    Raises KeyError if any required field is missing.
    Normalises confidence_level casing so rule conditions are case-insensitive.
    """
    per_structure = raw["per_structure_findings"]
    return Analysis(
        label=f"{raw['modality']} — {raw['impression']}",
        confidence=raw["confidence_level"].title(),   # "low" → "Low" defensive normalisation
        evidence=raw["key_findings"],
        findings=str(per_structure),
        recommendation=raw["recommendation"],
        metadata={
            "modality":         raw["modality"],
            "view":             raw["view"],
            "is_medical_image": raw["is_medical_image"],
            "per_structure":    per_structure,
            "severity_list":    [f["severity"] for f in per_structure],
        },
    )


# ── Rules ────────────────────────────────────────────────────────────────────


class NonMedicalImageRule(BaseRule):
    name = "non_medical_image_check"
    description = "Block analysis if the input is not a medical scan"

    def condition(self, analysis, context):
        return not analysis.metadata.get("is_medical_image", True)

    def action(self, analysis, context):
        analysis.label = "BLOCKED — Not a medical image"
        analysis.confidence = "Low"
        return analysis, RuleResult(
            action_taken=True,
            action_type="block",
            message="Input is not a medical image — analysis blocked.",
        )


class LowConfidenceRule(BaseRule):
    name = "low_confidence_flag"
    description = "Flag reports where model confidence is Low"

    def condition(self, analysis, context):
        # Guard: do not fire when NonMedicalImageRule already blocked this analysis.
        if not analysis.metadata.get("is_medical_image", True):
            return False
        return analysis.confidence == "Low"

    def action(self, analysis, context):
        analysis.recommendation += (
            " ⚠️ Low model confidence — senior radiologist review required."
        )
        return analysis, RuleResult(
            action_taken=True,
            action_type="flag",
            message="Low confidence: flagged for senior review.",
        )


class SeverityConsistencyRule(BaseRule):
    name = "severity_consistency_check"
    description = "Correct impression if critical findings contradict a 'normal' summary"

    def condition(self, analysis, context):
        has_critical = "critical" in analysis.metadata.get("severity_list", [])
        impression_normal = "normal" in analysis.label.lower()
        return has_critical and impression_normal

    def action(self, analysis, context):
        analysis.confidence = "Low"
        analysis.recommendation = (
            "Severity inconsistency detected — manual review required."
        )
        return analysis, RuleResult(
            action_taken=True,
            action_type="correct",
            message="Critical finding contradicts normal impression — confidence corrected to Low.",
        )


# ── Engine (module-level singleton) ──────────────────────────────────────────

engine = GuardrailEngine()
engine.register(NonMedicalImageRule())
engine.register(LowConfidenceRule())
engine.register(SeverityConsistencyRule())

# ── Pipeline factory ──────────────────────────────────────────────────────────


def build_pipeline(model) -> VLMGuardPipeline:
    """Create a VLMGuardPipeline that wraps the given Gemini model.

    The caller (streamlit_app.py) owns model creation and genai.configure().
    This function has no side effects on import — google.generativeai is not
    imported at module level so offline tests can import this module freely.
    """
    def gemini_model_fn(image: Image.Image, prompt: str) -> str:
        response = model.generate_content(
            [prompt, image],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": RADIOLOGY_JSON_SCHEMA,
            },
            safety_settings=SAFETY_SETTINGS,
        )
        return response.text

    return VLMGuardPipeline(
        model_fn=gemini_model_fn,
        parser_fn=parse_to_analysis,
        guardrail_engine=engine,
        enhancer_fn=ImageEnhancer(EnhancementStrategy.HIGH_CONTRAST),
    )
