import os

import google.generativeai as genai
import streamlit as st
from PIL import Image

from radiology_pipeline import build_pipeline

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MedVision Assist",
    page_icon="🩻",
    layout="wide",
)

st.warning("⚠️ RESEARCH PROTOTYPE ONLY. NOT FOR CLINICAL DIAGNOSIS.")
st.title("🩻 MedVision Assist")
st.markdown("**AI-Powered Radiology Triage** | _Powered by Google Gemini_")
st.markdown("---")

# ── API key + model setup ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")
    api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))

    if not api_key:
        st.error("⚠️ API Key missing.")
        st.info("Please set GOOGLE_API_KEY in Streamlit Secrets.")
        pipeline = None
    else:
        st.success("API Key Loaded")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=(
                "If this is not a medical image, set is_medical_image to false "
                "and leave per_structure_findings as an empty list."
            ),
        )
        pipeline = build_pipeline(model)

# ── Severity display helpers ──────────────────────────────────────────────────

_SEVERITY_ICONS = {
    "normal":   "🟢",
    "mild":     "🟡",
    "moderate": "🟠",
    "severe":   "🔴",
    "critical": "🔴🔴",
}

# ── Main layout ───────────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Scan")
    uploaded_file = st.file_uploader(
        "Drop X-Ray or CT Slice", type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Scan", use_column_width=True)
        analyze_clicked = st.button("Generate Preliminary Report", type="primary")

with col2:
    st.subheader("2. AI Analysis")

    if uploaded_file and analyze_clicked:
        if not api_key:
            st.error("Please configure the API Key first.")
        elif uploaded_file.size == 0:
            st.error("Empty file uploaded. Please upload a valid image.")
        else:
            with st.spinner("Analyzing anatomy and pathology..."):
                try:
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    result = pipeline.run(
                        image,
                        "Analyze this medical image.",
                        context={"scan_type": "radiology"},
                    )
                    validated = result.analysis
                    audit_entries = result.audit.summary()

                    was_blocked = any(
                        e["action"] == "block" for e in audit_entries
                    )

                    if was_blocked:
                        block_msg = next(
                            e["message"]
                            for e in audit_entries
                            if e["action"] == "block"
                        )
                        st.error(f"Analysis blocked: {block_msg}")
                        with st.expander("vlm-guard audit trail"):
                            for entry in audit_entries:
                                st.json(entry)
                        st.stop()

                    # ── Structured report ─────────────────────────────────

                    st.subheader(validated.label)

                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Modality", validated.metadata["modality"])
                        st.metric("View", validated.metadata["view"])
                    with m2:
                        st.metric("Confidence", validated.confidence)

                    st.markdown("**Findings:**")
                    for finding in validated.metadata["per_structure"]:
                        icon = _SEVERITY_ICONS.get(finding["severity"], "⚪")
                        st.markdown(
                            f"{icon} **{finding['structure']}**: "
                            f"{finding['observation']} ({finding['severity']})"
                        )

                    st.info(f"**Recommendation:** {validated.recommendation}")

                    with st.expander(
                        f"vlm-guard audit trail ({len(audit_entries)} rules fired)"
                    ):
                        if audit_entries:
                            for entry in audit_entries:
                                st.json(entry)
                        else:
                            st.success("No rules triggered — output passed all checks.")
                        st.caption(
                            "Validated by vlm-guard v0.1.2 — MohamedFakhry2007/vlm-guard"
                        )

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    elif not uploaded_file:
        st.info("Upload an image to see the analysis here.")
