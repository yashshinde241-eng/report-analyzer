"""
BEGINNER-FRIENDLY Medical Report Analyzer Backend
==================================================
This backend works immediately without any ML models!
Start here, then gradually add your real models.

Installation:
    pip install flask flask-cors pillow PyPDF2 python-docx --break-system-packages

Usage:
    python simple_backend.py

Then open report-analyzer.html in your browser!
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from PIL import Image
import PyPDF2
import docx
from io import BytesIO

app = Flask(__name__)
CORS(app)

# ============================================
# LOAD MODEL AT STARTUP
# ============================================
MODEL_PATH = r"C:\Users\LOQ\OneDrive\Desktop\report-analyzer\models\pneumonia_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🔧 Loading model on {device}...")

# Rebuild architecture
pneumonia_model = models.efficientnet_b0(weights=None)
num_features = pneumonia_model.classifier[1].in_features
pneumonia_model.classifier[1] = nn.Linear(num_features, 2)

# Load saved weights
checkpoint = torch.load(MODEL_PATH, map_location=device)
pneumonia_model.load_state_dict(checkpoint['model_state_dict'])
pneumonia_model = pneumonia_model.to(device)
pneumonia_model.eval()

# Image transform
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

print(f"✅ Model loaded successfully!")

# ============================================
# LOAD TEXT MODEL AT STARTUP
# ============================================
TEXT_MODEL_PATH = r"C:\Users\LOQ\OneDrive\Desktop\report-analyzer\models\text_model.pkl"
VECTORIZER_PATH = r"C:\Users\LOQ\OneDrive\Desktop\report-analyzer\models\vectorizer.pkl"

print(f"🔧 Loading text model...")
text_model = joblib.load(TEXT_MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
print(f"✅ Text model loaded successfully!")

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================
# ANALYSIS FUNCTIONS
# ============================================

def simple_file_analysis(text):
    """
    Real ML model inference using trained Logistic Regression
    """
    # Convert text to TF-IDF features
    text_tfidf = vectorizer.transform([text])

    # Get prediction
    prediction = text_model.predict(text_tfidf)[0]
    probability = text_model.predict_proba(text_tfidf)[0]
    confidence_score = round(max(probability) * 100, 2)
    detected = bool(prediction == 1)

    return {
        'detected': detected,
        'confidence': confidence_score,
        'diseaseName': 'Diabetes',
        'reportType': 'file'
    }

def simple_image_analysis(image):
    """
    Real ML model inference using trained EfficientNet
    """
    # Convert to RGB (handles grayscale X-rays too)
    image = image.convert('RGB')

    # Transform image
    image_tensor = inference_transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = pneumonia_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    class_names = ['NORMAL', 'PNEUMONIA']
    class_name = class_names[predicted_class.item()]
    confidence_score = round(confidence.item() * 100, 2)
    detected = class_name == 'PNEUMONIA'

    return {
        'detected': detected,
        'confidence': confidence_score,
        'diseaseName': 'Pneumonia',
        'reportType': 'image'
    }


# ============================================
# FILE PROCESSING FUNCTIONS
# ============================================

def extract_text_from_file(file):
    """Extract text from PDF, DOC, DOCX, TXT files"""
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
        print(f"❌ Error extracting text: {e}")
        return None


# ============================================
# API ENDPOINTS
# ============================================

@app.route('/api/analyze/file', methods=['POST'])
def analyze_file():
    """Analyze file-based medical report"""

    print("\n📄 Received file analysis request")

    # Check if file is present
    if 'report' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['report']

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    print(f"   File: {file.filename}")

    # Extract text
    text = extract_text_from_file(file)

    if text is None:
        return jsonify({'error': 'Failed to extract text from file'}), 400  # Fixed: 400 not 500

    print(f"   Extracted {len(text)} characters")

    # Analyze
    result = simple_file_analysis(text)

    print(f"   Result: {'✅ DETECTED' if result['detected'] else '❌ NOT DETECTED'}")
    print(f"   Confidence: {result['confidence']}%")

    return jsonify(result), 200


@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    """Analyze image-based medical report"""

    print("\n🖼️  Received image analysis request")

    # Check if file is present
    if 'report' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['report']

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    print(f"   Image: {file.filename}")

    try:
        # Load image
        image = Image.open(file)
        print(f"   Size: {image.size}, Format: {image.format}")

        # Analyze using real ML model
        result = simple_image_analysis(image)

        print(f"   Result: {'✅ DETECTED' if result['detected'] else '❌ NOT DETECTED'}")
        print(f"   Confidence: {result['confidence']}%")

        return jsonify(result), 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': f'Failed to analyze image: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'API is running',
        'model': 'EfficientNet-B0 (Pneumonia Detection)',
        'device': str(device)
    }), 200


@app.route('/', methods=['GET'])
def home():
    """Home endpoint with instructions"""
    return """
    <html>
    <head><title>Medical Report Analyzer API</title></head>
    <body style="font-family: Arial; padding: 40px; max-width: 800px; margin: 0 auto;">
        <h1>🏥 Medical Report Analyzer API</h1>
        <p><strong>Status:</strong> ✅ Running</p>
        <p><strong>Image Model:</strong> EfficientNet-B0 (Pneumonia Detection - 87.34% accuracy)</p>
        <p><strong>File Model:</strong> Keyword-based (Diabetes Detection)</p>

        <h2>Available Endpoints:</h2>
        <ul>
            <li><code>POST /api/analyze/file</code> - Analyze file reports</li>
            <li><code>POST /api/analyze/image</code> - Analyze image reports</li>
            <li><code>GET /api/health</code> - Health check</li>
        </ul>
    </body>
    </html>
    """


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🏥 MEDICAL REPORT ANALYZER - BACKEND")
    print("="*70)
    print("\n✅ Server starting...")
    print(f"🌐 URL: http://localhost:5000")
    print(f"🤖 Image Model: EfficientNet-B0 (Pneumonia Detection)")
    print(f"📄 File Model : Keyword-based (Diabetes Detection)")
    print("\n" + "="*70)
    print("Press CTRL+C to stop\n")

    app.run(debug=False, host='0.0.0.0', port=5000)