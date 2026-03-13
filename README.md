# Medical Report Analyzer

An AI-powered medical report analysis tool that detects **Diabetes** from clinical documents and **Pneumonia** from chest X-rays — with structured AI reasoning explaining every decision.

---

## What It Does

- Upload a medical report (PDF, DOCX, TXT) → detects diabetes using TF-IDF + Logistic Regression
- Upload a chest X-ray (JPG, PNG) → detects pneumonia using EfficientNet-B0
- After every analysis, Groq AI (Llama 3) generates a structured explanation with Summary, Key Evidence, Model Reasoning, and a Clinical Note

---

## Project Structure

```
report-analyzer/
├── report-analyzer.html      # Frontend (dark theme, purple accent)
├── simple_backend.py         # Flask backend — main server
├── test_backend.py           # API test script
├── test_model.py             # Standalone image model test
├── requirements.txt          # Python dependencies
├── .env                      # API keys — DO NOT commit
├── .env.example              # Template for .env
├── .gitignore
├── models/                   # Trained models — not in Git
│   ├── pneumonia_model.pth   # EfficientNet-B0 (87.34% accuracy)
│   ├── text_model.pkl        # Logistic Regression classifier
│   └── vectorizer.pkl        # TF-IDF vectorizer
└── data/
    └── text_reports.json     # Synthetic diabetes training data
```

---

## Models

| Model | Task | Architecture | Accuracy |
|-------|------|-------------|----------|
| Pneumonia Detection | Chest X-ray classification | EfficientNet-B0 | 87.34% |
| Diabetes Detection | Medical text classification | TF-IDF + Logistic Regression | 100%* |

*Trained on synthetic data — accuracy reflects training data quality.

**Training data:**
- Chest X-rays: [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — 5,216 images
- Diabetes reports: 1,000 synthetic clinical reports generated via `generate_text_data.py`

**Hardware used:** NVIDIA RTX 3050 6GB (CUDA 11.8, PyTorch 2.7)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/report-analyzer.git
cd report-analyzer
```

### 2. Install dependencies

```bash
pip install flask flask-cors pillow PyPDF2 python-docx torch torchvision joblib numpy requests python-dotenv
```

### 3. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your free Groq API key:
```
GROQ_API_KEY=your_groq_key_here
```

Get a free key at [console.groq.com](https://console.groq.com) — no credit card needed.

### 4. Add trained models

Place your trained model files in the `models/` folder:
```
models/pneumonia_model.pth
models/text_model.pkl
models/vectorizer.pkl
```

> Models are excluded from Git due to file size. Train them using `train_model.py` and `train_text_model.py`.

### 5. Run the backend

```bash
python simple_backend.py
```

Server starts at `http://localhost:5000`

### 6. Open the frontend

Open `report-analyzer.html` in your browser — no web server needed.

---

## Training the Models

### Pneumonia model (EfficientNet-B0)

1. Download the [Chest X-Ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle
2. Extract to `data set/chest_xray/`
3. Run:
```bash
python train_model.py
```

### Diabetes text model (TF-IDF + Logistic Regression)

1. Generate synthetic training data:
```bash
python generate_text_data.py
```
2. Train the classifier:
```bash
python train_text_model.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze/file` | Analyze a document report for diabetes |
| POST | `/api/analyze/image` | Analyze a chest X-ray for pneumonia |
| GET | `/api/health` | Check if the server is running |

**Request:** `multipart/form-data` with field name `report`

**Response:**
```json
{
  "detected": true,
  "confidence": 99.4,
  "diseaseName": "Pneumonia",
  "reportType": "image",
  "reasoning": "**Summary**\n..."
}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| Image Model | PyTorch, EfficientNet-B0 |
| Text Model | Scikit-learn, TF-IDF, Logistic Regression |
| AI Reasoning | Groq API (Llama 3.1 — free tier) |
| GPU | NVIDIA RTX 3050, CUDA 11.8 |

---

## Test Reports

Five sample diabetes reports are included for testing and showcase:

| File | Type | Expected Result |
|------|------|----------------|
| `01_Diabetes_Severe_Positive.docx` | HbA1c 9.8%, glucose 247 mg/dL | Detected |
| `02_Diabetes_Mild_Prediabetes.docx` | HbA1c 6.3%, glucose 118 mg/dL | Detected |
| `03_Diabetes_Negative_Normal.docx` | HbA1c 5.1%, glucose 88 mg/dL | Not detected |
| `04_Diabetes_Borderline_IGT.docx` | OGTT — impaired glucose tolerance | Detected |
| `05_Diabetes_Followup_Treatment.docx` | Known diabetic on Metformin | Detected |

---

## Disclaimer

This tool is for **educational purposes only**. It does not constitute medical advice or diagnosis. Always consult a qualified healthcare professional for proper evaluation and treatment.

---

## Academic Project

Built as a student project demonstrating the integration of machine learning models with a web interface for medical report analysis.