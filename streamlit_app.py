import os

import streamlit as st
from PIL import Image
# google.generativeai is imported lazily inside the Gemini branch so the offline
# Local CXR backend runs without the cloud SDK installed.

from radiology_pipeline import build_local_pipeline, build_pipeline

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MedVision Assist",
    page_icon="🩻",
    layout="wide",
)

st.warning("⚠️ RESEARCH PROTOTYPE ONLY. NOT FOR CLINICAL DIAGNOSIS.")
st.title("🩻 MedVision Assist")
st.markdown("**AI-Powered Radiology Triage** | _Gemini cloud or local CPU model_")
st.markdown("---")

# ── API key + model setup ─────────────────────────────────────────────────────


def _get_api_key():
    """Read GOOGLE_API_KEY from the environment, falling back to Streamlit
    secrets. Accessing st.secrets raises if no secrets.toml exists anywhere, so
    the lookup is guarded — a missing key returns None rather than crashing.
    """
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None


with st.sidebar:
    st.header("Configuration")

    backend = st.radio(
        "Analysis backend",
        ["Gemini (cloud)", "Local CXR (CPU)"],
        help=(
            "Gemini: any modality, needs GOOGLE_API_KEY. "
            "Local CXR: chest X-rays only, runs on CPU with no API key or network."
        ),
    )

    pipeline = None

    if backend == "Gemini (cloud)":
        api_key = _get_api_key()
        if not api_key:
            st.error("⚠️ API Key missing.")
            st.info("Please set GOOGLE_API_KEY in Streamlit Secrets.")
        else:
            try:
                import google.generativeai as genai
            except ImportError:
                st.error("google-generativeai is not installed.")
                st.code("pip install -r requirements.txt")
                genai = None

            if genai is not None:
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
    else:
        st.info("🖥️ Local chest-X-ray model — CPU only, no API key required.")
        st.caption("Chest radiographs only; other modalities are flagged, not analysed.")
        try:
            from local_backend import ensure_model_available

            ensure_model_available()
            pipeline = build_local_pipeline()
        except FileNotFoundError as e:
            st.error("Local model not found.")
            st.code(str(e))

# ── Severity display helpers ──────────────────────────────────────────────────

_SEVERITY_ICONS = {
    "normal":   "🟢",
    "mild":     "🟡",
    "moderate": "🟠",
    "severe":   "🔴",
    "critical": "🔴🔴",
}


def _render_finding(finding):
    icon = _SEVERITY_ICONS.get(finding["severity"], "⚪")
    st.markdown(
        f"{icon} **{finding['structure']}**: "
        f"{finding['observation']} ({finding['severity']})"
    )

# ── Main layout ───────────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Scan")
    uploaded_file = st.file_uploader(
        "Drop X-Ray or CT Slice", type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Scan", width="stretch")
        analyze_clicked = st.button("Generate Preliminary Report", type="primary")

with col2:
    st.subheader("2. AI Analysis")

    if uploaded_file and analyze_clicked:
        if pipeline is None:
            st.error("Selected backend is not ready. Check the sidebar configuration.")
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
                        modality = validated.metadata.get("modality", "")
                        if "unsupported" in modality.lower():
                            st.error(f"Analysis blocked — unsupported modality ({modality}).")
                        else:
                            st.error(f"Analysis blocked: {block_msg}")
                        # Surface the backend's specific guidance (e.g. CT → use Gemini).
                        if validated.recommendation:
                            st.info(validated.recommendation)
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

                    per_structure = validated.metadata["per_structure"]

                    if backend == "Local CXR (CPU)" and per_structure:
                        # Local findings are probability-ranked correlated labels:
                        # headline the strongest, fold the associated ones away.
                        st.markdown("**Primary finding:**")
                        _render_finding(per_structure[0])
                        associated = per_structure[1:]
                        if associated:
                            with st.expander(
                                f"Associated / differential findings ({len(associated)})"
                            ):
                                for finding in associated:
                                    _render_finding(finding)
                    else:
                        # Gemini returns distinct anatomy — show the flat list.
                        st.markdown("**Findings:**")
                        for finding in per_structure:
                            _render_finding(finding)

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
