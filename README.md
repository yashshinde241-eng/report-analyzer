# Medical Report Analyzer - Setup Guide

Complete guide to set up your medical report analyzer with backend integration.

---

## 📁 Project Structure

```
medical-report-analyzer/
├── frontend/
│   └── report-analyzer.html     # Your web interface
├── backend/
│   ├── flask_backend.py         # Flask backend option
│   ├── fastapi_backend.py       # FastAPI backend option
│   └── models/                  # Put your ML models here
│       ├── file_model.pkl
│       └── image_model.h5
└── README.md
```

---

## 🚀 Quick Start

### Option 1: Flask Backend (Recommended for beginners)

#### Step 1: Install Dependencies
```bash
pip install flask flask-cors pillow PyPDF2 python-docx
```

#### Step 2: Start the Server
```bash
python flask_backend.py
```

Server runs at: `http://localhost:5000`

#### Step 3: Update Frontend Configuration
In `report-analyzer.html`, line 415:
```javascript
const API_CONFIG = {
    BASE_URL: 'http://localhost:5000',
    // ...
};
```

---

### Option 2: FastAPI Backend (Better for production)

#### Step 1: Install Dependencies
```bash
pip install fastapi uvicorn python-multipart pillow PyPDF2 python-docx
```

#### Step 2: Start the Server
```bash
uvicorn fastapi_backend:app --reload --host 0.0.0.0 --port 8000
```

Server runs at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

#### Step 3: Update Frontend Configuration
In `report-analyzer.html`, line 415:
```javascript
const API_CONFIG = {
    BASE_URL: 'http://localhost:8000',
    // ...
};
```

---

## 🤖 Integrating Your ML Models

### For File Reports (Disease A)

**Flask** - Edit `flask_backend.py`, function `analyze_file_report()`:

```python
def analyze_file_report(file):
    text = extract_text_from_file(file)
    
    # Load your model
    import joblib  # or pickle, tensorflow, pytorch, etc.
    model = joblib.load('models/file_model.pkl')
    
    # Preprocess text
    # processed_text = your_preprocessing_function(text)
    
    # Make prediction
    prediction = model.predict([text])
    confidence = model.predict_proba([text])[0][1] * 100
    
    return {
        'detected': bool(prediction[0]),
        'confidence': int(confidence),
        'diseaseName': 'Disease A',
        'reportType': 'file'
    }
```

**FastAPI** - Edit `fastapi_backend.py`, function `analyze_file_report()`:

```python
async def analyze_file_report(file: UploadFile) -> Dict:
    text = await extract_text_from_file(file)
    
    # Load your model (load once at startup for better performance)
    import joblib
    model = joblib.load('models/file_model.pkl')
    
    # Preprocess and predict
    prediction = model.predict([text])
    confidence = model.predict_proba([text])[0][1] * 100
    
    return {
        'detected': bool(prediction[0]),
        'confidence': int(confidence),
        'diseaseName': 'Disease A',
        'reportType': 'file'
    }
```

---

### For Image Reports (Disease B)

**Flask** - Edit `flask_backend.py`, function `analyze_image_report()`:

```python
def analyze_image_report(file):
    image = Image.open(file)
    
    # Load your model
    from tensorflow import keras  # or pytorch, etc.
    model = keras.models.load_model('models/image_model.h5')
    
    # Preprocess image
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Make prediction
    prediction = model.predict(image_array)
    confidence = prediction[0][1] * 100
    
    return {
        'detected': bool(prediction[0][1] > 0.5),
        'confidence': int(confidence),
        'diseaseName': 'Disease B',
        'reportType': 'image'
    }
```

**FastAPI** - Similar pattern for `analyze_image_report()` function

---

## 🧪 Testing the Integration

### Test Backend Directly

**Flask:**
```bash
curl -X POST http://localhost:5000/api/health
```

**FastAPI:**
```bash
curl -X GET http://localhost:8000/api/health
```

### Test File Upload
```bash
curl -X POST -F "report=@test_file.pdf" http://localhost:5000/api/analyze/file
```

### Test from Frontend
1. Open `report-analyzer.html` in your browser
2. Upload a test file
3. Check browser console (F12) for request/response logs
4. Check backend terminal for incoming requests

---

## 🔧 Configuration Options

### Frontend Configuration (`report-analyzer.html`)

```javascript
const API_CONFIG = {
    // Development
    BASE_URL: 'http://localhost:5000',
    
    // Production
    // BASE_URL: 'https://your-domain.com',
    
    ENDPOINTS: {
        FILE_ANALYSIS: '/api/analyze/file',
        IMAGE_ANALYSIS: '/api/analyze/image'
    },
    
    // Optional: Add authentication
    AUTH_TOKEN: null,  // or 'Bearer your-token-here'
};
```

### Adding Authentication (Optional)

**Backend (Flask):**
```python
from functools import wraps
from flask import request

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or token != 'Bearer your-secret-token':
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/api/analyze/file', methods=['POST'])
@require_auth
def analyze_file():
    # ... your code
```

**Frontend:**
```javascript
const API_CONFIG = {
    AUTH_TOKEN: 'Bearer your-secret-token',
};
```

---

## 🌐 Deployment

### Deploy Backend to Cloud

**Heroku:**
```bash
# Create Procfile
echo "web: python flask_backend.py" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

**AWS/GCP/Azure:**
- Package your backend as a Docker container
- Deploy to cloud service (EC2, App Engine, Azure App Service)

### Update Frontend for Production
```javascript
const API_CONFIG = {
    BASE_URL: 'https://your-deployed-backend.com',
    // ...
};
```

---

## 🐛 Troubleshooting

### CORS Errors
**Error:** "Access-Control-Allow-Origin" blocked

**Solution:** Backend CORS is already configured, but if issues persist:
- Flask: Check `CORS(app)` is called
- FastAPI: Check `CORSMiddleware` configuration
- Ensure frontend and backend URLs match

### Connection Refused
**Error:** "Failed to fetch" or "Connection refused"

**Solutions:**
1. Ensure backend is running: Check terminal
2. Verify URL matches: Flask=5000, FastAPI=8000
3. Check firewall settings
4. Try `http://localhost` instead of `http://127.0.0.1`

### File Upload Fails
**Error:** "Invalid file format" or "No file provided"

**Solutions:**
1. Check file size < 10MB
2. Verify file extension is allowed
3. Check browser console for error details
4. Verify FormData is being sent correctly

### Model Loading Errors
**Error:** "FileNotFoundError" or "Model not found"

**Solutions:**
1. Ensure model file exists in correct path
2. Use absolute paths: `os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')`
3. Check model format matches library (joblib, pickle, h5, etc.)

---

## 📊 Expected Response Format

Your backend must return JSON with this structure:

```json
{
    "detected": true,          // boolean: true = disease detected
    "confidence": 92,          // number: 0-100
    "diseaseName": "Disease A" // string: name of disease
}
```

Optional fields:
```json
{
    "detected": true,
    "confidence": 92,
    "diseaseName": "Disease A",
    "reportType": "file",      // 'file' or 'image'
    "additionalInfo": "..."    // any extra information
}
```

---

## 📚 Next Steps

1. ✅ Choose Flask or FastAPI
2. ✅ Install dependencies
3. ✅ Start backend server
4. ✅ Test with curl or Postman
5. ✅ Integrate your ML models
6. ✅ Test with frontend
7. ✅ Deploy to production

---

## 💡 Tips

- **Development:** Use mock data first, then integrate real models
- **Testing:** Test backend independently before connecting frontend
- **Logging:** Add `print()` statements to debug request flow
- **Performance:** Load models once at startup, not per request
- **Security:** Add authentication for production deployment
- **Error Handling:** Always validate inputs and return clear errors

---

## 📞 Need Help?

Common issues and solutions are in the Troubleshooting section above.

For model-specific questions:
- TensorFlow: https://www.tensorflow.org/api_docs
- PyTorch: https://pytorch.org/docs
- Scikit-learn: https://scikit-learn.org/stable/documentation.html

---

Good luck with your medical report analyzer! 🚀
