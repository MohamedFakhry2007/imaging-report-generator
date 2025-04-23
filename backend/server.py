from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.responses import StreamingResponse
import uvicorn
import os
import logging
import asyncio
import base64
import google.generativeai as genai
from pathlib import Path
from PIL import Image
from io import BytesIO

# /backend 
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Google Generative AI
api_key = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=api_key)

# Load the Arabic style sample text
try:
    with open(ROOT_DIR / 'arabic_style_sample.txt', 'r', encoding='utf-8') as file:
        arabic_style_sample = file.read()
        logger.info("Successfully loaded Arabic style sample text")
except Exception as e:
    logger.error(f"Error loading Arabic style sample: {e}")
    arabic_style_sample = "فشل في تحميل النص المرجعي"

@app.get("/api")
async def root():
    return {"message": "مرحبا بك في واجهة برمجة التطبيق للقصص العربية"}

@app.post("/api/generate-story")
async def generate_story(file: UploadFile = File(...)):
    try:
        logger.info(f"Received image upload: {file.filename}")
        
        # Read and validate the image file
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="ملف فارغ")
        
        # Process the image using PIL
        try:
            image = Image.open(BytesIO(contents))
            logger.info(f"Image processed successfully: {image.format}, {image.size}")
            
            # Prepare image for Gemini
            buffered = BytesIO()
            image.save(buffered, format=image.format if image.format else "JPEG")
            img_bytes = buffered.getvalue()
            
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            raise HTTPException(status_code=400, detail="فشل في معالجة الصورة")

        # Configure the Gemini model
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Create system prompt with style reference
        system_prompt = f"""
        أنت كاتب قصص متميز باللغة العربية. 
        سأقدم لك صورة، وأريد منك أن تكتب قصة قصيرة مستوحاة منها.
        
        فيما يلي مثال على أسلوب الكتابة الذي أريدك أن تتبعه:
        
        {arabic_style_sample}
        
        اكتب قصة جديدة مستوحاة من الصورة المرفقة، ولكن باستخدام نفس أسلوب القصة المقدمة أعلاه من حيث النبرة وبناء الجملة والمفردات والإحساس العام للسرد.
        ضع نفسك في مكان راوي قصص متمرس. لا تصف الصورة فقط، بل استخدمها كمصدر إلهام لإنشاء قصة كاملة وجذابة.
        اكتب بالعربية الفصحى المعاصرة، وتأكد من الحفاظ على التسلسل المنطقي للأحداث.
        """
        
        # Generate content with the Gemini model
        try:
            # Log when starting the API call
            logger.info("Starting Gemini API call")
            start_time = asyncio.get_event_loop().time()
            
            # Create a multimodal prompt with the image and system prompt
            response = model.generate_content([system_prompt, {"mime_type": "image/jpeg", "data": img_bytes}])
            
            # Calculate and log API call latency
            latency = asyncio.get_event_loop().time() - start_time
            logger.info(f"Gemini API call completed in {latency:.2f} seconds")
            
            # Return the generated story
            return {"story": response.text}
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise HTTPException(status_code=500, detail=f"فشل في إنشاء القصة: {str(e)}")
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="حدث خطأ غير متوقع")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
