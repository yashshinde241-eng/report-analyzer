"""
simple_backend.py — Report Analyzer + Smart Triage Priority System
Flask backend with:
  - Original single-file analysis endpoints (unchanged)
  - NEW: /api/triage/upload-bulk  — bulk ingest + EfficientNet scoring
  - NEW: /api/triage/queue        — priority-sorted triage queue
"""

import os
import io
import sqlite3
import traceback
from datetime import datetime

import requests
import torch
import torchvision.transforms as transforms
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from torchvision import models

# ── Config ───────────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
DB_PATH       = "reports.db"
MODEL_DIR     = "models"
UPLOAD_FOLDER = "uploads"
PNEUMONIA_MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_model.pth")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB for bulk uploads

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ═══════════════════════════════════════════════════════════════════════════
# 1.  GROQ AI REASONING
# ═══════════════════════════════════════════════════════════════════════════

def get_ai_reasoning(disease_name: str, detected: bool, confidence: float,
                     report_type: str, extra_context: str = "") -> str:
    if not GROQ_API_KEY:
        return "AI reasoning unavailable — GROQ_API_KEY not set."
    try:
        status = "DETECTED" if detected else "NOT DETECTED"
        prompt = f"""You are a clinical AI assistant. Analyze this medical report result and provide structured reasoning.

Disease: {disease_name}
Status: {status}
Confidence: {confidence:.1%}
Report Type: {report_type}
{extra_context}

Provide a structured analysis with these exact sections:
**Summary:** (1-2 sentences)
**Key Evidence:** (bullet points)
**Model Reasoning:** (technical explanation)
**Clinical Note:** (important disclaimer)"""

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant", "max_tokens": 500,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI reasoning error: {str(e)}"

# ═══════════════════════════════════════════════════════════════════════════
# 2.  PNEUMONIA IMAGE MODEL  (EfficientNet-B0)
# ═══════════════════════════════════════════════════════════════════════════

pneumonia_model = None

def load_pneumonia_model():
    global pneumonia_model
    if pneumonia_model is not None:
        return pneumonia_model
    if not os.path.exists(PNEUMONIA_MODEL_PATH):
        raise FileNotFoundError(f"Pneumonia model not found at {PNEUMONIA_MODEL_PATH}")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    checkpoint = torch.load(PNEUMONIA_MODEL_PATH, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    pneumonia_model = model
    print("[INFO] Pneumonia model loaded.")
    return pneumonia_model


def predict_pneumonia(image_bytes: bytes) -> dict:
    """Run inference on raw image bytes. Returns prediction dict."""
    model = load_pneumonia_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    labels     = ["NORMAL", "PNEUMONIA"]
    prediction = labels[pred_idx.item()]
    confidence = conf.item()
    detected   = prediction == "PNEUMONIA"

    # Severity score: high when Pneumonia confident, high when Normal confidence is LOW
    severity_score = confidence if detected else (1.0 - confidence)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "detected":   detected,
        "severity_score": severity_score,
    }

# ═══════════════════════════════════════════════════════════════════════════
# 3.  DATABASE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_db():
    """Create the reports table if it doesn't exist yet."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            filename       TEXT    NOT NULL,
            status         TEXT    NOT NULL DEFAULT 'Pending'
                               CHECK(status IN ('Pending', 'Analyzed')),
            prediction     TEXT,
            confidence     REAL,
            severity_score REAL,
            timestamp      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def insert_pending(filename: str) -> int:
    conn = get_db()
    cur  = conn.execute(
        "INSERT INTO reports (filename, status) VALUES (?, 'Pending')", (filename,)
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def update_analyzed(row_id: int, prediction: str, confidence: float,
                    severity_score: float):
    conn = get_db()
    conn.execute("""
        UPDATE reports
        SET status='Analyzed', prediction=?, confidence=?, severity_score=?,
            timestamp=CURRENT_TIMESTAMP
        WHERE id=?
    """, (prediction, confidence, severity_score, row_id))
    conn.commit()
    conn.close()

# ═══════════════════════════════════════════════════════════════════════════
# 4.  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(device),
                    "groq_configured": bool(GROQ_API_KEY)})


@app.route("/api/analyze/image", methods=["POST"])
def analyze_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    try:
        image_bytes = file.read()
        result      = predict_pneumonia(image_bytes)
        reasoning   = get_ai_reasoning(
            "Pneumonia", result["detected"], result["confidence"],
            "Chest X-Ray (Image)")
        return jsonify({
            "disease": "Pneumonia",
            "prediction": result["prediction"],
            "confidence": round(result["confidence"] * 100, 2),
            "detected": result["detected"],
            "severity_score": round(result["severity_score"], 4),
            "reasoning": reasoning,
            "filename": file.filename,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════
# 6.  TRIAGE ENDPOINTS  (new)
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/triage/analyze-one", methods=["POST"])
def triage_analyze_one():
    """
    Analyze a SINGLE image and persist to DB. Called per-file by the frontend
    so the UI can update progressively as each result arrives.
    """
    ensure_db()
    if "file" not in request.files:
        return jsonify({"error": "No file field named 'file'"}), 400

    file     = request.files["file"]
    filename = file.filename or "unknown"
    row_id   = insert_pending(filename)

    try:
        image_bytes = file.read()
        if len(image_bytes) == 0:
            raise ValueError("Received empty file — check multipart upload")

        pred = predict_pneumonia(image_bytes)
        update_analyzed(row_id, pred["prediction"], pred["confidence"],
                        pred["severity_score"])

        severity = round(pred["severity_score"] * 100, 2)
        priority = "High" if severity >= 70 else ("Medium" if severity >= 40 else "Low")

        return jsonify({
            "id":             row_id,
            "filename":       filename,
            "status":         "Analyzed",
            "prediction":     pred["prediction"],
            "confidence":     round(pred["confidence"] * 100, 2),
            "severity_score": severity,
            "priority":       priority,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "id":       row_id,
            "filename": filename,
            "status":   "Error",
            "error":    str(e),
        }), 500


@app.route("/api/triage/upload-bulk", methods=["POST"])
def triage_upload_bulk():
    """
    Accept 1–20 chest X-ray images, run the pneumonia model on each,
    persist results to SQLite, return per-file results sorted by severity.
    """
    ensure_db()
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded. Use field name 'files'."}), 400
    if len(files) > 20:
        return jsonify({"error": "Maximum 20 files per batch."}), 400

    results = []
    errors  = []

    for file in files:
        filename = file.filename or "unknown"
        row_id   = insert_pending(filename)
        try:
            image_bytes    = file.read()
            pred           = predict_pneumonia(image_bytes)
            update_analyzed(row_id, pred["prediction"], pred["confidence"],
                            pred["severity_score"])
            results.append({
                "id":            row_id,
                "filename":      filename,
                "status":        "Analyzed",
                "prediction":    pred["prediction"],
                "confidence":    round(pred["confidence"] * 100, 2),
                "severity_score": round(pred["severity_score"] * 100, 2),
            })
        except Exception as e:
            errors.append({"filename": filename, "error": str(e)})

    # Sort by severity descending before returning
    results.sort(key=lambda x: x["severity_score"], reverse=True)

    return jsonify({
        "processed": len(results),
        "errors":    len(errors),
        "results":   results,
        "error_details": errors,
    })


@app.route("/api/triage/queue", methods=["GET"])
def triage_queue():
    """
    Return all analyzed reports sorted by severity_score descending.
    Optional query params:
      ?status=Pending|Analyzed|all  (default: all)
      ?limit=N                       (default: 100)
    """
    ensure_db()
    status = request.args.get("status", "all")
    limit  = int(request.args.get("limit", 100))

    conn  = get_db()
    if status == "all":
        rows = conn.execute(
            "SELECT * FROM reports ORDER BY severity_score DESC NULLS LAST LIMIT ?",
            (limit,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM reports WHERE status=? ORDER BY severity_score DESC NULLS LAST LIMIT ?",
            (status, limit)
        ).fetchall()
    conn.close()

    queue = []
    for r in rows:
        severity = r["severity_score"]
        if severity is None:
            priority = "Pending"
        elif severity >= 0.70:
            priority = "High"
        elif severity >= 0.40:
            priority = "Medium"
        else:
            priority = "Low"

        queue.append({
            "id":            r["id"],
            "filename":      r["filename"],
            "status":        r["status"],
            "prediction":    r["prediction"],
            "confidence":    round(r["confidence"] * 100, 2) if r["confidence"] else None,
            "severity_score": round(r["severity_score"] * 100, 2) if r["severity_score"] else None,
            "priority":      priority,
            "timestamp":     r["timestamp"],
        })

    return jsonify({"total": len(queue), "queue": queue})


@app.route("/api/triage/clear", methods=["DELETE"])
def triage_clear():
    """Delete all records — useful for testing."""
    ensure_db()
    conn = get_db()
    conn.execute("DELETE FROM reports")
    conn.commit()
    conn.close()
    return jsonify({"message": "Triage queue cleared."})

# ═══════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ensure_db()
    print("[INFO] Starting Report Analyzer + Triage System on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
