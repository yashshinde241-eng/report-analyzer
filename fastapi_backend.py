"""
FastAPI Backend for Medical Report Analyzer
============================================
This is a complete FastAPI backend that integrates with your frontend.

Installation:
    pip install fastapi uvicorn python-multipart pillow PyPDF2 python-docx --break-system-packages

Usage:
    uvicorn fastapi_backend:app --reload --host 0.0.0.0 --port 8000

The server will run on http://localhost:8000
API docs available at: http://localhost:8000/docs
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import PyPDF2
import docx
from io import BytesIO
import random
from typing import Dict

app = FastAPI(
    title="Medical Report Analyzer API",
    description="API for analyzing medical reports from files and images",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
ALLOWED_FILE_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def validate_file_extension(filename: str, allowed_extensions: set) -> bool:
    """Validate file extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


async def extract_text_from_file(file: UploadFile) -> str:
    """Extract text from various file formats"""
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    content = await file.read()
    
    try:
        if file_ext == 'txt':
            return content.decode('utf-8')
        
        elif file_ext == 'pdf':
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        
        elif file_ext in ['doc', 'docx']:
            doc = docx.Document(BytesIO(content))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")


async def analyze_file_report(file: UploadFile) -> Dict:
    """
    Analyze file-based medical report
    
    TODO: Replace this with your actual ML model for file analysis
    This is where you'd load your trained model and make predictions
    """
    # Extract text from file
    text = await extract_text_from_file(file)
    
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from file")
    
    # TODO: Load your ML model and make prediction
    # Example:
    # from your_model import FileReportClassifier
    # model = FileReportClassifier.load('path/to/model.pkl')
    # prediction = model.predict(text)
    # confidence = model.predict_proba(text)
    
    # Mock prediction for demonstration
    # Replace this with your actual model inference
    detected = random.choice([True, False])
    confidence = random.randint(85, 98)
    disease_name = "Disease A"  # Replace with your disease name
    
    return {
        'detected': detected,
        'confidence': confidence,
        'diseaseName': disease_name,
        'reportType': 'file',
        'extractedTextLength': len(text)
    }


async def analyze_image_report(file: UploadFile) -> Dict:
    """
    Analyze image-based medical report
    
    TODO: Replace this with your actual ML model for image analysis
    This is where you'd load your trained model and make predictions
    """
    try:
        # Read and process image
        content = await file.read()
        image = Image.open(BytesIO(content))
        
        # TODO: Preprocess image for your model
        # Example preprocessing:
        # image = image.resize((224, 224))
        # image_array = np.array(image) / 255.0
        
        # TODO: Load your ML model and make prediction
        # Example:
        # from your_model import ImageReportClassifier
        # model = ImageReportClassifier.load('path/to/model.h5')
        # prediction = model.predict(image_array)
        # confidence = prediction[0][1] * 100
        
        # Mock prediction for demonstration
        # Replace this with your actual model inference
        detected = random.choice([True, False])
        confidence = random.randint(85, 98)
        disease_name = "Disease B"  # Replace with your disease name
        
        return {
            'detected': detected,
            'confidence': confidence,
            'diseaseName': disease_name,
            'reportType': 'image',
            'imageSize': f"{image.width}x{image.height}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")


@app.post("/api/analyze/file")
async def analyze_file_endpoint(report: UploadFile = File(...)):
    """Endpoint for file-based report analysis"""
    
    # Validate file
    if not report.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not validate_file_extension(report.filename, ALLOWED_FILE_EXTENSIONS):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Allowed: PDF, DOC, DOCX, TXT"
        )
    
    # Analyze the report
    result = await analyze_file_report(report)
    return JSONResponse(content=result, status_code=200)


@app.post("/api/analyze/image")
async def analyze_image_endpoint(report: UploadFile = File(...)):
    """Endpoint for image-based report analysis"""
    
    # Validate file
    if not report.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not validate_file_extension(report.filename, ALLOWED_IMAGE_EXTENSIONS):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Allowed: JPG, JPEG, PNG"
        )
    
    # Analyze the image
    result = await analyze_image_report(report)
    return JSONResponse(content=result, status_code=200)


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "API is running",
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Medical Report Analyzer API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "file_analysis": "/api/analyze/file",
            "image_analysis": "/api/analyze/image",
            "health": "/api/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("Medical Report Analyzer - FastAPI Backend")
    print("=" * 60)
    print("Server running on: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\nAvailable endpoints:")
    print("  POST /api/analyze/file  - Analyze file reports")
    print("  POST /api/analyze/image - Analyze image reports")
    print("  GET  /api/health        - Health check")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
