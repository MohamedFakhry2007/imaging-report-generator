# 🩻 MedVision Assist (Radiology AI Prototype)

### Overview

**MedVision Assist** is a specialized Clinical Decision Support System (CDSS) prototype designed to assist radiologists in drafting preliminary imaging reports.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://imaging-report-generator-zg9lmrthbghzlrk2fefu48.streamlit.app)

> **Try the Live Demo:** [Click here to use the app in your browser](https://imaging-report-generator-zg9lmrthbghzlrk2fefu48.streamlit.app)

The app ships as a **Streamlit** application with two interchangeable analysis backends, selectable from the sidebar:

  * **Gemini (cloud)** — Google's multimodal `gemini-2.0-flash`; handles any modality, needs an API key.
  * **Local CXR (CPU)** — a ~28 MB [TorchXRayVision](https://github.com/mlmed/torchxrayvision) classifier exported to ONNX. Runs on cheap CPUs with **no API key and no network**, fits the Streamlit Cloud free tier, but is **chest-X-ray only**.

Either way, output is validated and guard-railed by [vlm-guard](https://github.com/MohamedFakhry2007/vlm-guard) into a structured, schema-checked report, with a "Human-in-the-loop" workflow for clinical safety.

Click the image to watch the demo:

[![Watch the Demo](https://img.youtube.com/vi/4NDVZmV-ls4/maxresdefault.jpg)](https://youtu.be/4NDVZmV-ls4)

### ✨ Key Features

#### 1\. Multimodal AI Analysis

  * **Visual Question Answering (VQA):** Utilizes Gemini to "see" medical anomalies in unstructured image data.
  * **Anatomy Recognition:** Automatically identifies scan modality (e.g., "Chest X-Ray, PA View") and key structures.
  * **Pathology Detection:** Highlights potential findings such as opacities, fractures, or effusions based on visual patterns.

#### 2\. Structured Reporting

  * **Standardized Output:** Generates reports in a strict clinical format: **Modality → Findings → Impression**.
  * **Zero-Shot Prompting:** Uses advanced prompt engineering to eliminate hallucinations and adhere to professional radiological tone.
  * **Draft-First Approach:** Designed to speed up reporting workflows by providing a 90% complete draft for physician review.

#### 3\. Clinical Safety & Privacy

  * **Stateless Processing:** Images are processed in memory and never stored permanently, ensuring compliance with data minimization principles.
  * **Disclaimer Injection:** Every generated report includes mandatory safety disclaimers to prevent misuse.
  * **Filtered Inference:** Custom safety thresholds optimized for medical imagery (distinguishing organs from gore).

#### 4\. Modern Web Interface

  * **Drag-and-Drop Upload:** Intuitive UI for rapid image processing.
  * **Real-time Analysis:** visual feedback during the AI inference stage.
  * **Responsive Design:** Fully functional on desktop and tablet devices for portable ward usage.

-----

### 🛠️ Technical Architecture

  * **Python 3.9+** with **Streamlit** for the UI (`streamlit_app.py`).
  * **vlm-guard:** schema validation + guardrail rules (non-medical block, low-confidence flag, severity-consistency correction) in `radiology_pipeline.py`.
  * **Backend A — Gemini:** `google-generativeai` driving `gemini-2.0-flash` with a constrained JSON response schema.
  * **Backend B — Local CXR:** `TorchXRayVision` DenseNet exported to ONNX (`tools/export_onnx.py`), run on CPU via **onnxruntime** + **numpy** in `local_backend.py`. No PyTorch at run time.
  * **Pillow (PIL):** image preprocessing and the `looks_like_xray` gate.

> **Legacy full-stack:** a FastAPI + React.js version lives in `backend/` and `frontend/`. It is no longer the primary entry point — the Streamlit app above is what is deployed and maintained.

-----

### 🚀 Quick Start

This runs the Streamlit app locally. You only need an API key if you want to use the **Gemini** backend — the **Local CXR** backend runs offline on CPU.

#### Prerequisites

  * **Python 3.9+**
  * *(Gemini backend only)* a **Google AI Studio API Key** — get one [here](https://aistudio.google.com/).

#### 1. Clone and install

```bash
git clone https://github.com/MohamedFakhry2007/imaging-report-generator.git
cd imaging-report-generator

# Create + activate a virtual environment
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# Install runtime dependencies (Streamlit, vlm-guard, onnxruntime, numpy, Pillow)
pip install -r requirements.txt
```

#### 2. Run the app

```bash
streamlit run streamlit_app.py
```

This opens **`http://localhost:8501`** in your browser. Pick a backend from the **sidebar** — then follow the matching setup below.

#### 3a. Using the Gemini backend (cloud)

Provide your API key one of two ways, then select **"Gemini (cloud)"** in the sidebar:

```bash
# Option 1 — environment variable (PowerShell)
$env:GOOGLE_API_KEY = "your_actual_api_key_here"
streamlit run streamlit_app.py
```

```toml
# Option 2 — .streamlit/secrets.toml (persists across runs)
GOOGLE_API_KEY = "your_actual_api_key_here"
```

#### 3b. Using the Local CXR backend (CPU, no API key)

Select **"Local CXR (CPU)"** in the sidebar. This needs the ONNX model at `models/chexnet.onnx`. If that file isn't already present, generate it **once**:

```bash
# Heavy, dev-only deps (PyTorch + torchxrayvision) — NOT needed to run the app
pip install -r requirements-export.txt
python tools/export_onnx.py        # downloads weights, writes models/chexnet.onnx (~28 MB)
```

After the model exists, the app runs it with onnxruntime alone — no PyTorch, no network.

-----

### 📖 Usage Guide

1.  **Choose a backend:** In the sidebar, pick **Gemini (cloud)** or **Local CXR (CPU)**.
2.  **Upload:** Drag a Chest X-Ray or CT slice (JPG/PNG) into the drop zone.
3.  **Analyze:** Click **"Generate Preliminary Report"**.
4.  **Review:** A structured, guard-railed report appears in the right panel. Non-chest-X-ray uploads on the Local backend are flagged as unsupported rather than analysed.

-----

### 📋 Project Structure

```text
imaging-report-generator/
├── streamlit_app.py          # ▶ Main app entry point (run with: streamlit run)
├── radiology_pipeline.py     # vlm-guard pipeline: schema, rules, parser, backends
├── local_backend.py          # Local CPU chest-X-ray backend (ONNX inference)
├── requirements.txt          # Runtime deps (Streamlit, vlm-guard, onnxruntime, numpy)
├── requirements-export.txt   # Dev-only deps for the ONNX export (PyTorch)
├── models/
│   └── chexnet.onnx          # Exported classifier (~28 MB; generated by the script)
├── tools/
│   └── export_onnx.py        # One-time TorchXRayVision → ONNX export
├── tests/                    # Offline pytest suite (no API key / model required)
├── backend/                  # Legacy FastAPI server (not the primary entry point)
├── frontend/                 # Legacy React UI (not the primary entry point)
└── README.md                 # Documentation
```

-----

### 🐛 Troubleshooting

**"streamlit: command not found"**

  * Ensure the virtual environment is active (`.\venv\Scripts\Activate.ps1`) and you ran `pip install -r requirements.txt`. You can also use `python -m streamlit run streamlit_app.py`.

**"API Key missing" (Gemini backend)**

  * Set `GOOGLE_API_KEY` as an environment variable or in `.streamlit/secrets.toml`. On Streamlit Cloud, add it under **App → Settings → Secrets**.

**"Local CXR model not found" / `models/chexnet.onnx` missing**

  * Generate it once: `pip install -r requirements-export.txt` then `python tools/export_onnx.py`.

**PowerShell blocks the activate script**

  * Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, then activate again.

-----

### 📄 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

### ⚠️ Disclaimer

**Research Prototype Only:** This tool is intended for educational and research purposes to demonstrate the capabilities of Large Multimodal Models (LMMs) in healthcare. It is **not** a certified medical device and should **not** be used for primary diagnosis or patient care.

-----

### 👨‍💻 Author

**Mohamed Fakhry**

  * **Role:** Clinical AI Engineer
  * **GitHub:** [@MohamedFakhry2007](https://github.com/MohamedFakhry2007)
  * **Email:** mohamedfakhrysmile@gmail.com

### 🙏 Support

If you find this project interesting for the Clinical AI space, please consider:

  * ⭐ **Starring** the repository
  * 💡 **Sharing** it with your network
  * 🤝 **Contributing** to improve the prompt engineering
