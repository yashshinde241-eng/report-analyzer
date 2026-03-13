"""
Medical Report Analyzer Backend
================================
EfficientNet-B0      - Pneumonia Detection
TF-IDF + LogReg      - Diabetes Detection
Groq API (FREE)      - Human-readable reasoning (Llama 3)
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
import joblib
import numpy as np
import requests
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
# GROQ API CONFIG  (100% Free)
# ============================================

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.1-8b-instant"   # free, fast, great for reasoning


def generate_reasoning(prompt):
    """
    Call Groq API (free) using Llama 3.
    Falls back to a rule-based explanation if the API call fails.
    """
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json"
        }
        body = {
            "model":       GROQ_MODEL,
            "max_tokens":  600,
            "temperature": 0.3,
            "messages": [
                {
                    "role":    "system",
                    "content": "You are a medical AI assistant that explains machine learning model decisions clearly and professionally. Always follow the exact format requested."
                },
                {
                    "role":    "user",
                    "content": prompt
                }
            ]
        }
        response = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=15)
        response.raise_for_status()
        data = response.json()
        result = data["choices"][0]["message"]["content"].strip()
        print(f"   Groq reasoning OK ({len(result)} chars)")
        return result

    except Exception as e:
        print(f"Groq API error: {type(e).__name__}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response body: {e.response.text}")
        return None


# ============================================
# LOAD IMAGE MODEL AT STARTUP
# ============================================
MODEL_PATH = r"C:\Users\LOQ\OneDrive\Desktop\report-analyzer\models\pneumonia_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🔧 Loading image model on {device}...")

pneumonia_model = models.efficientnet_b0(weights=None)
num_features = pneumonia_model.classifier[1].in_features
pneumonia_model.classifier[1] = nn.Linear(num_features, 2)

checkpoint = torch.load(MODEL_PATH, map_location=device)
pneumonia_model.load_state_dict(checkpoint['model_state_dict'])
pneumonia_model = pneumonia_model.to(device)
pneumonia_model.eval()

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

print("✅ Image model loaded!")

# ============================================
# LOAD TEXT MODEL AT STARTUP
# ============================================
TEXT_MODEL_PATH = r"C:\Users\LOQ\OneDrive\Desktop\report-analyzer\models\text_model.pkl"
VECTORIZER_PATH = r"C:\Users\LOQ\OneDrive\Desktop\report-analyzer\models\vectorizer.pkl"

print("🔧 Loading text model...")
text_model = joblib.load(TEXT_MODEL_PATH)
vectorizer  = joblib.load(VECTORIZER_PATH)
print("✅ Text model loaded!")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================
# ANALYSIS FUNCTIONS
# ============================================

def simple_file_analysis(text):
    """
    TF-IDF + LogReg prediction, then Groq (Llama 3) writes structured reasoning
    based on the top keywords that influenced the decision.
    """
    text_tfidf  = vectorizer.transform([text])
    prediction  = text_model.predict(text_tfidf)[0]
    probability = text_model.predict_proba(text_tfidf)[0]
    confidence  = round(float(max(probability)) * 100, 2)
    detected    = bool(prediction == 1)

    # Extract top keywords
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores  = text_tfidf.toarray()[0]
    coef          = text_model.coef_[0]
    combined      = tfidf_scores * coef
    top_indices   = np.argsort(np.abs(combined))[::-1][:12]

    top_keywords = [
        feature_names[idx]
        for idx in top_indices
        if tfidf_scores[idx] > 0
    ]

    # Build prompt for Groq
    result_label = "POSITIVE (Diabetes Detected)" if detected else "NEGATIVE (No Diabetes Detected)"
    keywords_str = ", ".join(top_keywords) if top_keywords else "no specific keywords found"

    prompt = f"""You are a medical AI assistant helping explain a machine learning model's decision.

A TF-IDF + Logistic Regression classifier analyzed a medical text report and produced the following result:

- Result: {result_label}
- Confidence: {confidence}%
- Top influential terms found in the report: {keywords_str}
- Report excerpt (first 800 chars): {text[:800]}

Write a detailed, structured reasoning explanation for why the model made this decision. Format your response with these exact sections:

**Summary**
One sentence stating the overall finding.

**Key Evidence**
3-4 bullet points listing the specific terms or phrases from the report that most strongly influenced the decision, and briefly why each one is clinically relevant to diabetes.

**Model Reasoning**
2-3 sentences explaining how the combination of evidence led to this conclusion.

**Clinical Note**
One sentence reminding that this is an AI prediction and not a medical diagnosis.

Keep the tone professional and educational. Do not repeat the confidence score."""

    reasoning_text = generate_reasoning(prompt)

    if not reasoning_text:
        reasoning_text = f"**Summary**\nThe model classified this report as {result_label}.\n\n**Key Evidence**\n- Terms found: {keywords_str}\n\n**Model Reasoning**\nThe classifier identified these terms as statistically associated with diabetes based on training data.\n\n**Clinical Note**\nThis is an AI prediction only - consult a healthcare professional."

    return {
        'detected':    detected,
        'confidence':  confidence,
        'diseaseName': 'Diabetes',
        'reportType':  'file',
        'reasoning':   reasoning_text
    }


def simple_image_analysis(image):
    """
    EfficientNet-B0 prediction, then Groq (Llama 3) writes structured reasoning
    based on confidence scores and clinical knowledge of pneumonia.
    """
    image        = image.convert('RGB')
    img_w, img_h = image.size
    image_tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs       = pneumonia_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    prob_normal    = round(float(probabilities[0].item()) * 100, 1)
    prob_pneumonia = round(float(probabilities[1].item()) * 100, 1)
    detected       = prob_pneumonia > prob_normal
    confidence     = prob_pneumonia if detected else prob_normal

    # Build prompt for Groq
    result_label = "POSITIVE (Pneumonia Detected)" if detected else "NEGATIVE (No Pneumonia Detected)"

    prompt = f"""You are a medical AI assistant helping explain a deep learning model's chest X-ray analysis.

An EfficientNet-B0 model (trained on 5,216 chest X-ray images, 87.34% test accuracy) analyzed a chest X-ray and produced:

- Result: {result_label}
- Pneumonia probability: {prob_pneumonia}%
- Normal probability: {prob_normal}%
- Image dimensions: {img_w}x{img_h} pixels

Write a detailed, structured reasoning explanation for this result. Format your response with these exact sections:

**Summary**
One sentence stating the overall finding.

**Key Indicators**
3-4 bullet points describing what visual patterns in chest X-rays typically lead to this classification (e.g. opacity, consolidation, clear lung fields), and how the confidence score reflects the model's certainty.

**Model Reasoning**
2-3 sentences explaining how EfficientNet-B0 arrives at this type of decision and what the probability split ({prob_normal}% Normal vs {prob_pneumonia}% Pneumonia) indicates about the scan.

**Clinical Note**
One sentence reminding that this is an AI prediction and not a radiologist's diagnosis.

Keep the tone professional and educational."""

    reasoning_text = generate_reasoning(prompt)

    if not reasoning_text:
        status = "detected pneumonia" if detected else "found no signs of pneumonia"
        reasoning_text = f"**Summary**\nThe model {status} with {confidence}% confidence.\n\n**Key Indicators**\n- Normal probability: {prob_normal}%\n- Pneumonia probability: {prob_pneumonia}%\n\n**Model Reasoning**\nEfficientNet-B0 analyzed the pixel patterns in the chest X-ray and classified it based on features learned from 5,216 training images.\n\n**Clinical Note**\nThis is an AI prediction only - consult a radiologist for proper diagnosis."

    return {
        'detected':    detected,
        'confidence':  confidence,
        'diseaseName': 'Pneumonia',
        'reportType':  'image',
        'reasoning':   reasoning_text
    }


# ============================================
# FILE PROCESSING
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
            text = "\n".join([p.text for p in doc.paragraphs])
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
    print("\n📄 Received file analysis request")

    if 'report' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['report']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    print(f"   File: {file.filename}")

    text = extract_text_from_file(file)
    if text is None:
        return jsonify({'error': 'Failed to extract text from file'}), 400

    print(f"   Extracted {len(text)} characters")
    print("   Generating reasoning via Groq API...")

    result = simple_file_analysis(text)

    print(f"   Result: {'✅ DETECTED' if result['detected'] else '❌ NOT DETECTED'}")
    print(f"   Confidence: {result['confidence']}%")

    return jsonify(result), 200


@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    print("\n🖼️  Received image analysis request")

    if 'report' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['report']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    print(f"   Image: {file.filename}")

    try:
        image = Image.open(file)
        print(f"   Size: {image.size}, Format: {image.format}")
        print("   Generating reasoning via Groq API...")

        result = simple_image_analysis(image)

        print(f"   Result: {'✅ DETECTED' if result['detected'] else '❌ NOT DETECTED'}")
        print(f"   Confidence: {result['confidence']}%")

        return jsonify(result), 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': f'Failed to analyze image: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status':    'healthy',
        'message':   'API is running',
        'models':    'EfficientNet-B0 + TF-IDF LogReg',
        'reasoning': 'Groq API - Llama 3 (free)',
        'device':    str(device)
    }), 200


@app.route('/', methods=['GET'])
def home():
    return """
    <html>
    <head><title>Medical Report Analyzer API</title></head>
    <body style="font-family: Arial; padding: 40px; max-width: 800px; margin: 0 auto;">
        <h1>&#127973; Medical Report Analyzer API</h1>
        <p><strong>Status:</strong> &#9989; Running</p>
        <p><strong>Image Model:</strong> EfficientNet-B0 (Pneumonia &mdash; 87.34% accuracy)</p>
        <p><strong>Text Model:</strong> TF-IDF + Logistic Regression (Diabetes)</p>
        <p><strong>Reasoning:</strong> Groq API - Llama 3 (free)</p>
        <h2>Endpoints:</h2>
        <ul>
            <li><code>POST /api/analyze/file</code> &mdash; Analyze document reports</li>
            <li><code>POST /api/analyze/image</code> &mdash; Analyze X-ray images</li>
            <li><code>GET /api/health</code> &mdash; Health check</li>
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
    print(f"🤖 Image Model : EfficientNet-B0 (Pneumonia)")
    print(f"📄 Text Model  : TF-IDF + Logistic Regression (Diabetes)")
    print(f"🧠 Reasoning   : Groq API - Llama 3 (FREE)")
    print("\n" + "="*70)
    print("Press CTRL+C to stop\n")

    app.run(debug=False, host='0.0.0.0', port=5000)