"""
Flask Backend for Medical Report Analyzer
==========================================
This is a complete Flask backend that integrates with your frontend.

Installation:
    pip install flask flask-cors pillow PyPDF2 python-docx --break-system-packages

Usage:
    python flask_backend.py

The server will run on http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from PIL import Image
import PyPDF2
import docx
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_FILE_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def extract_text_from_file(file):
    """Extract text from various file formats"""
    filename = secure_filename(file.filename)
    file_ext = filename.rsplit('.', 1)[1].lower()
    
    try:
        if file_ext == 'txt':
            return file.read().decode('utf-8')
        
        elif file_ext == 'pdf':
            pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        
        elif file_ext in ['doc', 'docx']:
            doc = docx.Document(BytesIO(file.read()))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        
        else:
            return None
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None


def analyze_file_report(file):
    """
    Analyze file-based medical report
    
    TODO: Replace this with your actual ML model for file analysis
    This is where you'd load your trained model and make predictions
    """
    # Extract text from file
    text = extract_text_from_file(file)
    
    if text is None:
        return None
    
    # TODO: Load your ML model and make prediction
    # Example:
    # from your_model import FileReportClassifier
    # model = FileReportClassifier.load('path/to/model.pkl')
    # prediction = model.predict(text)
    # confidence = model.predict_proba(text)
    
    # Mock prediction for demonstration
    # Replace this with your actual model inference
    import random
    detected = random.choice([True, False])
    confidence = random.randint(85, 98)
    disease_name = "Disease A"  # Replace with your disease name
    
    return {
        'detected': detected,
        'confidence': confidence,
        'diseaseName': disease_name,
        'reportType': 'file'
    }


def analyze_image_report(file):
    """
    Analyze image-based medical report
    
    TODO: Replace this with your actual ML model for image analysis
    This is where you'd load your trained model and make predictions
    """
    try:
        # Load and process image
        image = Image.open(file)
        
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
        import random
        detected = random.choice([True, False])
        confidence = random.randint(85, 98)
        disease_name = "Disease B"  # Replace with your disease name
        
        return {
            'detected': detected,
            'confidence': confidence,
            'diseaseName': disease_name,
            'reportType': 'image'
        }
    
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None


@app.route('/api/analyze/file', methods=['POST'])
def analyze_file():
    """Endpoint for file-based report analysis"""
    
    # Check if file is present
    if 'report' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['report']
    
    # Check if file has a name
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    # Validate file extension
    if not allowed_file(file.filename, ALLOWED_FILE_EXTENSIONS):
        return jsonify({'error': 'Invalid file format. Allowed: PDF, DOC, DOCX, TXT'}), 400
    
    # Analyze the report
    result = analyze_file_report(file)
    
    if result is None:
        return jsonify({'error': 'Failed to analyze report'}), 500
    
    return jsonify(result), 200


@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    """Endpoint for image-based report analysis"""
    
    # Check if file is present
    if 'report' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['report']
    
    # Check if file has a name
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    # Validate file extension
    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({'error': 'Invalid file format. Allowed: JPG, JPEG, PNG'}), 400
    
    # Analyze the image
    result = analyze_image_report(file)
    
    if result is None:
        return jsonify({'error': 'Failed to analyze image'}), 500
    
    return jsonify(result), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'API is running'}), 200


if __name__ == '__main__':
    print("=" * 50)
    print("Medical Report Analyzer - Flask Backend")
    print("=" * 50)
    print("Server running on: http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  POST /api/analyze/file  - Analyze file reports")
    print("  POST /api/analyze/image - Analyze image reports")
    print("  GET  /api/health        - Health check")
    print("\nPress CTRL+C to stop the server")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
