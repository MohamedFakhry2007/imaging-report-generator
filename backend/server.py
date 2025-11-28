from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
import uvicorn
import os
import logging
import asyncio
import google.generativeai as genai
from pathlib import Path
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Load Environment
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

app = FastAPI(title="Medical Imaging Report Assistant")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION START ---

# 1. Google GenAI Config
api_key = os.environ.get('GOOGLE_API_KEY')
if not api_key:
    logger.error("GOOGLE_API_KEY not found. Please check your .env file.")
    
genai.configure(api_key=api_key)

# 2. System Prompt
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

# 3. Model Initialization
# We use the string that works for you. 
try:
    model = genai.GenerativeModel(
        model_name='gemini-2.0-flash', # Switched to 2.0 as per your request
        system_instruction=SYSTEM_PROMPT
    )
    logger.info("Gemini Model initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Gemini Model: {e}")

# 4. Safety Settings (Block None to allow medical imagery)
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

# Database (Optional)
mongo_url = os.environ.get('MONGO_URL')
client = None
db = None
if mongo_url:
    try:
        client = AsyncIOMotorClient(mongo_url)
        db = client[os.environ.get('DB_NAME', 'med_vision_db')]
        logger.info("Connected to MongoDB.")
    except Exception as e:
        logger.warning(f"Could not connect to MongoDB: {e}")

# --- CONFIGURATION END ---

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "mode": "clinical-prototype"}

@app.post("/api/generate-report")
async def generate_report(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        
        # 1. Read and Validate File
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        try:
            # Load directly into PIL Image
            image = Image.open(BytesIO(contents))
            
            # Convert to RGB (Critical for X-rays)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            logger.info(f"Image processed: {image.format} {image.size} mode={image.mode}")
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # 2. Generate Content (ASYNC)
        # We use await here so the server doesn't freeze while thinking
        response = await model.generate_content_async(
            ["Analyze this medical image.", image], 
            safety_settings=SAFETY_SETTINGS
        )
        
        if not response.text:
             logger.error("Model returned empty response.")
             raise HTTPException(status_code=500, detail="AI Model returned empty response.")

        # 3. Log to DB (FIXED)
        # We await this directly to avoid the 'Future vs Coroutine' crash
        if db is not None:
            try:
                await db.reports.insert_one({
                    "filename": file.filename,
                    "report_length": len(response.text),
                    "timestamp": asyncio.get_event_loop().time()
                })
            except Exception as db_e:
                logger.error(f"Failed to log to DB: {db_e}")
                # We do not raise here, so the user still gets their report even if DB fails

        return {"report": response.text}

    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"detail": f"Processing error: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)