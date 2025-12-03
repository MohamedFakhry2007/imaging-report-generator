# 🩻 MedVision Assist (Radiology AI Prototype)

### Overview

**MedVision Assist** is a specialized Clinical Decision Support System (CDSS) prototype designed to assist radiologists in drafting preliminary imaging reports.

Leveraging the multimodal capabilities of **Google Gemini**, this application bridges the gap between raw medical imaging and structured textual analysis. It automates the generation of findings for X-Rays and CT scans, enforcing standardized reporting formats (RSNA-style) while maintaining a "Human-in-the-loop" workflow to ensure clinical safety.

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

#### Backend

  * **Python 3.9+**
  * **FastAPI:** High-performance async API framework.
  * **Google Generative AI:** Gemini (Multimodal) via API.
  * **Pillow (PIL):** Image preprocessing and validation.
  * **Security:** Environment-based configuration, CORS protection, and input sanitization.

#### Frontend

  * **React.js 18**
  * **Tailwind CSS:** Utility-first styling for a clean, medical-grade UI.
  * **Fetch API:** Asynchronous communication with the inference engine.

-----

### 🚀 Quick Start

#### Prerequisites

  * **Python 3.9+**
  * **Node.js 18+** (LTS Recommended)
  * **Google AI Studio API Key** (Get one [here](https://aistudio.google.com/))

#### 1\. Clone the Repository

```bash
git clone https://github.com/MohamedFakhry2007/imaging-report-generator.git
```

#### 2\. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GOOGLE_API_KEY=your_actual_api_key_here" > .env
```

#### 3\. Frontend Setup

Open a **new terminal** window and navigate to the frontend folder.

```bash
cd frontend

# Install Node dependencies
npm install

# Create .env file for API connection
echo "REACT_APP_BACKEND_URL=http://localhost:8000" > .env
```

#### 4\. Run the Application

**Terminal 1 (Backend):**

```bash
# Ensure venv is active
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 (Frontend):**

```bash
npm start
```

Visit **`http://localhost:3000`** to access the application.

-----

### 📖 Usage Guide

1.  **Upload:** Drag a Chest X-Ray or CT slice (JPG/PNG) into the drop zone.
2.  **Analyze:** Click **"Generate Preliminary Report"**.
3.  **Review:** The AI will generate a structured report on the right panel within 3-5 seconds.
4.  **Reset:** Click "Remove Scan" to clear the session and start over.

-----

### 📋 Project Structure

```text
MedVision-Assist/
├── backend/                # Python FastAPI Server
│   ├── server.py           # Main application entry point
│   ├── requirements.txt    # Python dependencies
│   ├── .env                # API Keys (Not tracked in git)
│   └── venv/               # Virtual Environment
├── frontend/               # React Application
│   ├── public/             # Static assets
│   ├── src/
│   │   ├── App.js          # Main UI Logic
│   │   ├── App.css         # Component styles
│   │   └── index.js        # React DOM entry
│   ├── package.json        # Node dependencies
│   ├── tailwind.config.js  # CSS Configuration
│   └── .env                # Frontend config
└── README.md               # Documentation
```

-----

### 🐛 Troubleshooting

**"FastAPI/Uvicorn not found"**

  * Ensure you activated the virtual environment (`.\venv\Scripts\Activate`) before running the server.

**"npm command not found"**

  * Ensure Node.js is installed and added to your system PATH.
  * On Windows PowerShell, you may need to run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

**"GoogleGenerativeAI Error"**

  * Check your `backend/.env` file. Ensure `GOOGLE_API_KEY` is pasted correctly without quotes or spaces.

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
