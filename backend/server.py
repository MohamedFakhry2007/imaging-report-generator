# backend/server.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import uvicorn
import os
import logging
import asyncio
import base64
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

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Google GenAI Config
api_key = os.environ.get('GOOGLE_API_KEY')
if not api_key:
    logger.error("GOOGLE_API_KEY not found")
genai.configure(api_key=api_key)

# Database (Optional - keep if you want to log usage)
mongo_url = os.environ.get('MONGO_URL')
if mongo_url:
    client = AsyncIOMotorClient(mongo_url)
    db = client[os.environ.get('DB_NAME', 'med_vision_db')]

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "mode": "clinical-prototype"}

@app.post("/api/generate-report")  # Renamed endpoint
async def generate_report(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing medical image: {file.filename}")
        
        # 1. Image Validation & Processing
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        try:
            image = Image.open(BytesIO(contents))
            # Convert to RGB if necessary (some X-rays are Grayscale/L mode)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            base64_encoded_image = base64.b64encode(img_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise HTTPException(status_code=400, detail="Invalid image format")

        # 2. Model Configuration
        # Use Gemini 1.5 Flash or Pro (Flash is faster for real-time inference)
        model = genai.GenerativeModel('gemini-2.0-flash') 
        
        # 3. Clinical System Prompt (The most important part)
        system_prompt = """
        You are an expert Radiologist Assistant AI. 
        Your task is to analyze the provided medical image and generate a structured preliminary report.
        
        Strictly follow this reporting format:
        
        **1. Modality:** (e.g., Chest X-ray, MRI, CT Scan)
        **2. Orientation/View:** (e.g., PA View, Lateral)
        **3. Key Findings:** - List objective observations.
           - Mention structures (Lungs, Heart, Bones, etc.).
           - Note any anomalies (opacity, fractures, effusion).
        **4. Impression:** A concise summary of the findings.
        
        **DISCLAIMER:** - If the image is NOT a medical image, reply: "Invalid input: This does not appear to be a medical image."
        - Always end the report with: "DISCLAIMER: This is an AI-generated prototype for research purposes only. Not for clinical diagnosis."
        """
        
        # 4. Multimodal Generation
        image_part = {"inline_data": {"mime_type": "image/jpeg", "data": base64_encoded_image}}
        text_part = {"text": system_prompt}
        
        # Adjust safety settings for medical imagery (organs/bones are not "violence")
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
        ]

        response = model.generate_content(
            contents=[text_part, image_part],
            safety_settings=safety_settings
        )
        
        if not response.text:
             raise HTTPException(status_code=500, detail="Model returned empty response")

        return {"report": response.text}

    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))