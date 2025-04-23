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
            
            # Ensure the image is in a supported format (JPEG/PNG)
            buffered = BytesIO()
            save_format = image.format if image.format in ['JPEG', 'PNG'] else 'JPEG'
            image.save(buffered, format=save_format)
            img_bytes = buffered.getvalue()
            
            # Encode the image in base64 for the Gemini API
            base64_encoded_image = base64.b64encode(img_bytes).decode('utf-8')
            logger.info("Image encoded successfully in base64")
            
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            raise HTTPException(status_code=400, detail="فشل في معالجة الصورة")

        # Configure the Gemini model for multimodal input (image + text)
        # Using a known working multimodal model as of March 2025
        model = genai.GenerativeModel('gemini-1.5-pro-001')
        logger.info("Using Gemini 1.5 Pro model for multimodal input")
        
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
            
            # Prepare multimodal content parts according to latest Gemini API specs
            image_part = {"inline_data": {"mime_type": "image/jpeg", "data": base64_encoded_image}}
            text_part = {"text": system_prompt}
            
            # Set safety settings to be less restrictive but still safe
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            # Create a multimodal prompt with the image and system prompt
            response = model.generate_content(
                contents=[text_part, image_part],
                safety_settings=safety_settings
            )
            
            # Calculate and log API call latency
            latency = asyncio.get_event_loop().time() - start_time
            logger.info(f"Gemini API call completed in {latency:.2f} seconds")
            
            # Check if the response contains blocked content
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 'SAFETY':
                    logger.warning("Content blocked due to safety filters")
                    raise HTTPException(status_code=400, detail="تم حظر المحتوى بسبب قواعد السلامة")
            
            # Check if we have text content
            if hasattr(response, 'text'):
                return {"story": response.text}
            else:
                # Try to extract text from candidates if available
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                return {"story": part.text}
                
                # If we still don't have text, return a fallback message
                return {"story": "عفواً، لم أتمكن من إنشاء قصة من هذه الصورة. يرجى تجربة صورة أخرى."}
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Gemini API error: {error_msg}")
            
            # Check specifically for safety filter blocks
            if "safety_ratings" in error_msg or "finish_reason" in error_msg:
                return {"story": "عفواً، لم أتمكن من إنشاء قصة من هذه الصورة بسبب قيود المحتوى. يرجى تجربة صورة أخرى."}
            
            # More specific error messages based on common API errors
            if "API key" in error_msg:
                raise HTTPException(status_code=500, detail="فشل في إنشاء القصة: مفتاح API غير صالح")
            elif "quota" in error_msg.lower():
                raise HTTPException(status_code=500, detail="فشل في إنشاء القصة: تم تجاوز الحصة المسموح بها")
            elif "blocked" in error_msg.lower() or "content" in error_msg.lower():
                raise HTTPException(status_code=500, detail="فشل في إنشاء القصة: المحتوى محظور")
            else:
                raise HTTPException(status_code=500, detail="فشل في إنشاء القصة")
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="حدث خطأ غير متوقع")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
