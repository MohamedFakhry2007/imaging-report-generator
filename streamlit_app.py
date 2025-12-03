import streamlit as st
import google.generativeai as genai
from PIL import Image
import os

# 1. Page Configuration
st.set_page_config(
    page_title="MedVision Assist",
    page_icon="🩻",
    layout="wide"
)

# 2. Disclaimer Banner (Ported from frontend/src/App.js)
st.warning("⚠️ RESEARCH PROTOTYPE ONLY. NOT FOR CLINICAL DIAGNOSIS.")

# 3. Header
st.title("🩻 MedVision Assist")
st.markdown("**AI-Powered Radiology Triage** | _Powered by Google Gemini_")
st.markdown("---")

# 4. Sidebar: API Key Configuration
# This allows the app to work if you set the key in Streamlit Secrets, 
# or allows a user to input their own if they clone it.
with st.sidebar:
    st.header("Configuration")
    # Try to get key from secrets first, otherwise look for env variable
    api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))
    
    if not api_key:
        st.error("⚠️ API Key missing.")
        st.info("Please set GOOGLE_API_KEY in Streamlit Secrets.")
    else:
        st.success("API Key Loaded")
        genai.configure(api_key=api_key)

# 5. Logic: System Prompt (Ported from backend/server.py)
SYSTEM_PROMPT = """
You are an expert Radiologist Assistant AI. 
Your task is to analyze the provided medical image and generate a structured preliminary report.

Strictly follow this reporting format:
**1. Modality:** (e.g., Chest X-ray, MRI, CT Scan)
**2. Orientation/View:** (e.g., PA View, Lateral)
**3. Key Findings:** - List objective observations.
   - Mention structures (Lungs, Heart, Bones, etc.).
   - Note any anomalies (opacity, fractures, effusion).
**4. Impression:** A concise summary of the findings.

**IMPORTANT DISCLAIMERS:**
- If the image is NOT a medical image, reply: "Invalid input: This does not appear to be a medical image."
- Always end the report with: "DISCLAIMER: This is an AI-generated prototype for research purposes only. Not for clinical diagnosis."
"""

# 6. Model Setup
# Using the model defined in your backend
model = genai.GenerativeModel(
    model_name='gemini-2.0-flash', 
    system_instruction=SYSTEM_PROMPT
)

# 7. UI: Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Scan")
    uploaded_file = st.file_uploader("Drop X-Ray or CT Slice", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Scan", use_column_width=True)
        
        # Analyze Button
        analyze_clicked = st.button("Generate Preliminary Report", type="primary")

with col2:
    st.subheader("2. AI Analysis")
    
    if uploaded_file and analyze_clicked:
        if not api_key:
            st.error("Please configure the API Key first.")
        else:
            with st.spinner("Analyzing anatomy and pathology..."):
                try:
                    # Convert to RGB (Logic from backend/server.py)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                        
                    # Generate Content
                    response = model.generate_content(
                        ["Analyze this medical image.", image]
                    )
                    
                    st.markdown("### Findings")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    
    elif not uploaded_file:
        st.info("Upload an image to see the analysis here.")