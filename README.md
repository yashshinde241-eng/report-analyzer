# Medical Report Analyzer

An AI-powered chest X-ray analysis tool that detects **Pneumonia** using a privacy-preserving local inference pipeline — with structured AI reasoning explaining every decision.

---

## What It Does

- Upload a chest X-ray (JPG, PNG) → detects pneumonia using EfficientNet-B0 with TTA
- Local vision model runs entirely on-premises — **zero images leave the server**
- Only anonymised numeric metrics are sent to Groq AI (Llama 3.1) for clinical reasoning
- Smart Triage Dashboard for batch X-ray processing with priority queuing
- Federated Control Tower for coordinating distributed hospital node training

---

## Project Structure

```
report-analyzer/
├── report-analyzer.html      # Frontend — 3-panel SPA (Analysis, Triage, Fed. Tower)
├── simple_backend.py         # Flask backend — main server + SSE + FL orchestrator
├── xray_transforms.py        # CLAHE preprocessing, Albumentations, TTA inference
├── federated_client.py       # Simulated hospital FL node
├── federated_server.py       # Standalone FL server CLI
├── data_splitter.py          # Splits dataset across hospital nodes
├── train_model.py            # EfficientNet-B0 training script
├── test_backend.py           # API test script (6 tests)
├── test_model.py             # Standalone image model test
├── init_db.py                # Database initialisation
├── requirements.txt          # Python dependencies
├── .env                      # API keys — DO NOT commit
├── .env.example              # Template for .env
├── .gitignore
├── models/                   # Trained models — not in Git
│   └── pneumonia_model.pth   # EfficientNet-B0 (87.34% accuracy)
├── data/
│   ├── hospital_A/           # Federated node data split
│   ├── hospital_B/
│   └── hospital_C/
└── uploads/                  # Temporary upload storage
```

---

## Model

| Model | Task | Architecture | Accuracy |
|-------|------|-------------|----------|
| Pneumonia Detection | Chest X-ray classification | EfficientNet-B0 | 87.34% |

**Training data:** [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — 5,216 images

**Hardware used:** NVIDIA RTX 3050 6GB (CUDA 11.8, PyTorch 2.7)

**Robustness techniques applied during training:**
- CLAHE preprocessing
- Albumentations augmentation (ElasticTransform, GridDistortion, GaussNoise, CoarseDropout)
- Label smoothing (0.1)
- Cosine annealing warm restarts
- Mixup (α=0.4)
- Test-Time Augmentation (TTA) — 4-pass fast inference

---

## Privacy-Preserving Pipeline

```
[Raw image bytes] → local EfficientNet-B0 TTA → numeric metrics
                                                      │
                                                      ▼
                                         anonymised text summary
                                    (no pixel data, no patient info)
                                                      │
                                                      ▼
                                         Groq API (text only)
                                         Llama-3.1-8b-instant
                                                      │
                                                      ▼
                                         clinical reasoning text
```

**Guarantee:** `image_sent_externally: false` — raw bytes never leave the Flask process.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/report-analyzer.git
cd report-analyzer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
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

### 4. Add trained model

Place your trained model file in the `models/` folder:
```
models/pneumonia_model.pth
```

> Model is excluded from Git due to file size. Train it using `train_model.py`.

### 5. Run the backend

```bash
python simple_backend.py
```

Server starts at `http://localhost:5000`

### 6. Open the frontend

Open `report-analyzer.html` in your browser — no web server needed.

---

## Training the Model

1. Download the [Chest X-Ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle
2. Extract to `data set/chest_xray/`
3. Run:
```bash
python train_model.py
```

---

## Federated Learning (Optional)

Run 3 simulated hospital nodes alongside the main backend:

```bash
# Terminal 1 — Main backend
python simple_backend.py

# Run once to split data
python data_splitter.py

# Terminals 2–4 — Hospital nodes
python federated_client.py 5001 data/hospital_A
python federated_client.py 5002 data/hospital_B
python federated_client.py 5003 data/hospital_C
```

Then use the **Federated Control Tower** tab in the frontend to trigger training rounds and monitor nodes in real time via SSE.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze/image` | Analyze a chest X-ray for pneumonia |
| POST | `/api/triage/analyze-one` | Analyze a single X-ray for the triage queue |
| POST | `/api/triage/upload-bulk` | Batch analyze up to 20 X-rays |
| GET | `/api/triage/queue` | Retrieve the triage queue |
| DELETE | `/api/triage/clear` | Clear the triage queue |
| GET | `/api/federated/stream` | SSE stream for FL round events |
| POST | `/api/federated/trigger` | Start a federated training round |
| GET | `/api/federated/nodes` | Live health status of all FL nodes |
| GET | `/api/health` | Check server status |

**Response example (`/api/analyze/image`):**
```json
{
  "disease": "Pneumonia",
  "prediction": "PNEUMONIA",
  "confidence": 94.2,
  "detected": true,
  "severity_score": 82.1,
  "normal_prob": 5.8,
  "pneumonia_prob": 94.2,
  "tta_passes": 4,
  "privacy": {
    "image_sent_externally": false,
    "anonymised_summary": "LOCAL_ANALYSIS: Class=PNEUMONIA, ...",
    "external_service": "Groq / Llama-3.1-8b-instant"
  },
  "reasoning": "• Triage urgency: ..."
}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, JavaScript (Vanilla), Chart.js |
| Backend | Python, Flask, Flask-CORS |
| Image Model | PyTorch, EfficientNet-B0, TTA |
| Augmentation | Albumentations, CLAHE |
| AI Reasoning | Groq API (Llama 3.1 — free tier) |
| Database | SQLite (triage queue persistence) |
| Federated Learning | FedAvg, SSE streaming |
| GPU | NVIDIA RTX 3050, CUDA 11.8 |

---

## Disclaimer

This tool is for **educational purposes only**. It does not constitute medical advice or diagnosis. Always consult a qualified healthcare professional for proper evaluation and treatment.

---

## Academic Project

Built as a student project demonstrating privacy-preserving medical AI, federated learning, and real-time web interfaces.
