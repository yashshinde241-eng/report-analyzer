"""
simple_backend.py — Report Analyzer + Smart Triage Priority System
====================================================================
Techniques integrated (Phase 1.5)
----------------------------------
  1. CLAHE preprocessing     — applied inside run_tta_inference (xray_transforms)
  2. Albumentations pipeline — all transforms now in xray_transforms module
  3. Label smoothing / cosine annealing — training-time; no backend change needed
  4. TTA inference           — predict_pneumonia() now calls run_tta_inference()
     Default: 4-pass TTA ("fast").  Pass tta_level="full" for 8-pass.

Endpoints
---------
  GET  /api/health                — liveness probe (reports TTA config)
  POST /api/analyze/image         — single chest X-ray analysis with TTA
  POST /api/triage/analyze-one    — analyze + persist to SQLite
  POST /api/triage/upload-bulk    — batch analyze up to 20 images
  GET  /api/triage/queue          — priority-sorted triage queue
  DELETE /api/triage/clear        — wipe triage queue

All severity/confidence values are on a 0–100 float scale throughout.
"""

import io
import os
import sqlite3
import traceback
from pathlib import Path
from datetime import datetime

import requests
import torch
import torch.nn as nn
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from torchvision import models

# Import the shared transform / TTA / CLAHE module (Techniques 1, 2, 4)
from xray_transforms import run_tta_inference

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT_DIR             = Path(__file__).resolve().parent
MODEL_DIR            = ROOT_DIR / "models"
UPLOAD_DIR           = ROOT_DIR / "uploads"
DB_PATH              = ROOT_DIR / "reports.db"
PNEUMONIA_MODEL_PATH = MODEL_DIR / "pneumonia_model.pth"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(ROOT_DIR / ".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ── TTA configuration ─────────────────────────────────────────────────────────
# "fast" = 4 augmented passes (~40 ms/image on CPU).
# "full" = 8 passes  (~80 ms/image on CPU) — use for offline/batch analysis.
TTA_LEVEL = os.getenv("TTA_LEVEL", "fast")   # override via .env if needed
print(f"[INFO] TTA level: {TTA_LEVEL}")

# ── Allowed upload types ──────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_MIMETYPES  = {"image/png", "image/jpeg"}


def _validate_image_file(file) -> str | None:
    """
    Validate file extension and MIME type before any processing.
    Returns an error string if invalid, None if acceptable.
    """
    if not file or not file.filename:
        return "No file provided."
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return f"File extension '.{ext}' is not allowed. Use PNG or JPEG."
    mime = (file.mimetype or "").lower()
    if mime not in ALLOWED_MIMETYPES:
        return f"MIME type '{mime}' is not allowed. Expected image/png or image/jpeg."
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# GROQ AI REASONING
# ═══════════════════════════════════════════════════════════════════════════════

def get_ai_reasoning(
    disease_name: str,
    detected: bool,
    confidence: float,      # 0–1 for the prompt
    report_type: str,
    tta_passes: int = 1,
    extra_context: str = "",
) -> str:
    """
    Query the Groq Llama API for structured clinical reasoning.
    Returns a plain-text explanation or a graceful fallback message.
    """
    if not GROQ_API_KEY:
        return "AI reasoning unavailable — GROQ_API_KEY not configured in .env."
    try:
        status = "DETECTED" if detected else "NOT DETECTED"
        prompt = (
            f"You are a clinical AI assistant. Analyse this medical report result "
            f"and provide structured reasoning.\n\n"
            f"Disease    : {disease_name}\n"
            f"Status     : {status}\n"
            f"Confidence : {confidence:.1%}\n"
            f"Report Type: {report_type}\n"
            f"Method     : Test-Time Augmentation ({tta_passes} passes averaged)\n"
            f"{extra_context}\n\n"
            f"Provide a structured analysis with these exact sections:\n"
            f"**Summary:** (1-2 sentences)\n"
            f"**Key Evidence:** (bullet points)\n"
            f"**Model Reasoning:** (technical explanation)\n"
            f"**Clinical Note:** (important disclaimer)"
        )
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model":      "llama-3.1-8b-instant",
                "max_tokens": 500,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"AI reasoning error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
# PNEUMONIA IMAGE MODEL  (EfficientNet-B0)
# ═══════════════════════════════════════════════════════════════════════════════

_pneumonia_model = None   # module-level singleton


def load_pneumonia_model() -> nn.Module:
    """
    Load the EfficientNet-B0 pneumonia classifier.
    If the checkpoint is missing, initialise with random weights and warn loudly.
    The server stays functional but predictions will be meaningless until a
    real checkpoint is placed at models/pneumonia_model.pth.
    """
    global _pneumonia_model
    if _pneumonia_model is not None:
        return _pneumonia_model

    model = models.efficientnet_b0(weights=None)

    # Replicate the regularised head from train_model.py
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.2),
        nn.Linear(512, 2),
    )

    if PNEUMONIA_MODEL_PATH.exists():
        checkpoint  = torch.load(
            str(PNEUMONIA_MODEL_PATH),
            map_location=device,
            weights_only=True,
        )
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)
        techniques = checkpoint.get("techniques", []) if isinstance(checkpoint, dict) else []
        print("[INFO] Pneumonia model loaded.")
        if techniques:
            print(f"[INFO] Trained with: {', '.join(techniques)}")
    else:
        print(
            "\n" + "=" * 70 + "\n"
            "  WARNING: models/pneumonia_model.pth NOT FOUND.\n"
            "  Model initialised with RANDOM weights — predictions unreliable.\n"
            "  Place a trained checkpoint at:\n"
            f"    {PNEUMONIA_MODEL_PATH}\n"
            + "=" * 70 + "\n"
        )

    model.to(device)
    model.eval()
    _pneumonia_model = model
    return _pneumonia_model


def predict_pneumonia(image_bytes: bytes) -> dict:
    """
    Run TTA inference on raw image bytes.

    Pipeline (Techniques 1, 2, 4)
    ------------------------------
    1. CLAHE preprocessing   (inside run_tta_inference via xray_transforms)
    2. N albumentations views (4 fast / 8 full TTA passes)
    3. Average softmax across all passes
    4. Return structured dict — all metrics on 0–100 scale

    Returns
    -------
    dict with keys:
        prediction    : "NORMAL" | "PNEUMONIA"
        confidence    : float 0–100
        detected      : bool
        severity_score: float 0–100
        tta_passes    : int
        normal_prob   : float 0–100
        pneumonia_prob: float 0–100
    """
    model = load_pneumonia_model()
    return run_tta_inference(model, image_bytes, device, tta_level=TTA_LEVEL)


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_db() -> None:
    """Create the reports table if it does not already exist."""
    conn = _get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            filename       TEXT    NOT NULL,
            status         TEXT    NOT NULL DEFAULT 'Pending'
                               CHECK(status IN ('Pending', 'Analyzed')),
            prediction     TEXT,
            confidence     REAL,
            severity_score REAL,
            tta_passes     INTEGER,
            timestamp      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def insert_pending(filename: str) -> int:
    conn   = _get_db()
    cur    = conn.execute(
        "INSERT INTO reports (filename, status) VALUES (?, 'Pending')", (filename,)
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def update_analyzed(
    row_id:        int,
    prediction:    str,
    confidence:    float,        # 0–100
    severity_score: float,       # 0–100
    tta_passes:    int   = 1,
) -> None:
    """Persist analysis results — all numeric metrics on 0–100 scale."""
    conn = _get_db()
    conn.execute(
        """
        UPDATE reports
        SET status='Analyzed',
            prediction=?,
            confidence=?,
            severity_score=?,
            tta_passes=?,
            timestamp=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (prediction, confidence, severity_score, tta_passes, row_id),
    )
    conn.commit()
    conn.close()


def _priority_label(severity_score: float | None) -> str:
    """Map a 0–100 severity score to a triage priority label."""
    if severity_score is None:
        return "Pending"
    if severity_score >= 70:
        return "High"
    if severity_score >= 40:
        return "Medium"
    return "Low"


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":           "ok",
        "device":           str(device),
        "groq_configured":  bool(GROQ_API_KEY),
        "model_checkpoint": PNEUMONIA_MODEL_PATH.exists(),
        "tta_level":        TTA_LEVEL,
        "tta_passes":       4 if TTA_LEVEL == "fast" else 8,
    })


@app.route("/api/analyze/image", methods=["POST"])
def analyze_image():
    """
    Single chest X-ray analysis with CLAHE + TTA.
    Multipart form field: 'file' (PNG or JPEG).
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field named 'file' in request."}), 400

    file = request.files["file"]
    err  = _validate_image_file(file)
    if err:
        return jsonify({"error": err}), 400

    try:
        image_bytes = file.read()
        result      = predict_pneumonia(image_bytes)   # TTA, all metrics 0–100

        reasoning = get_ai_reasoning(
            "Pneumonia",
            result["detected"],
            result["confidence"] / 100,   # pass as 0–1 for % formatting in prompt
            "Chest X-Ray",
            tta_passes=result["tta_passes"],
        )

        return jsonify({
            "disease":       "Pneumonia",
            "prediction":    result["prediction"],
            "confidence":    result["confidence"],       # 0–100
            "detected":      result["detected"],
            "severity_score": result["severity_score"],  # 0–100
            "normal_prob":    result["normal_prob"],      # 0–100
            "pneumonia_prob": result["pneumonia_prob"],   # 0–100
            "tta_passes":     result["tta_passes"],
            "reasoning":      reasoning,
            "filename":       file.filename,
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ─── Triage endpoints ─────────────────────────────────────────────────────────

@app.route("/api/triage/analyze-one", methods=["POST"])
def triage_analyze_one():
    """
    Analyze one image with TTA and persist to the database.
    Multipart form field: 'file'.
    """
    ensure_db()

    if "file" not in request.files:
        return jsonify({"error": "No file field named 'file'."}), 400

    file = request.files["file"]
    err  = _validate_image_file(file)
    if err:
        return jsonify({"error": err}), 400

    filename = file.filename or "unknown"
    row_id   = insert_pending(filename)

    try:
        image_bytes = file.read()
        if not image_bytes:
            raise ValueError("Received empty file.")

        pred = predict_pneumonia(image_bytes)   # all metrics 0–100
        update_analyzed(
            row_id,
            pred["prediction"],
            pred["confidence"],
            pred["severity_score"],
            pred["tta_passes"],
        )

        return jsonify({
            "id":             row_id,
            "filename":       filename,
            "status":         "Analyzed",
            "prediction":     pred["prediction"],
            "confidence":     pred["confidence"],       # 0–100
            "severity_score": pred["severity_score"],   # 0–100
            "normal_prob":    pred["normal_prob"],       # 0–100
            "pneumonia_prob": pred["pneumonia_prob"],    # 0–100
            "tta_passes":     pred["tta_passes"],
            "priority":       _priority_label(pred["severity_score"]),
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({
            "id":       row_id,
            "filename": filename,
            "status":   "Error",
            "error":    str(exc),
        }), 500


@app.route("/api/triage/upload-bulk", methods=["POST"])
def triage_upload_bulk():
    """
    Batch analyze 1–20 X-rays with TTA, persist all, return sorted results.
    Multipart form field: 'files' (multiple).
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

        val_err = _validate_image_file(file)
        if val_err:
            errors.append({"filename": filename, "error": val_err})
            continue

        try:
            image_bytes = file.read()
            pred = predict_pneumonia(image_bytes)   # all metrics 0–100
            update_analyzed(
                row_id,
                pred["prediction"],
                pred["confidence"],
                pred["severity_score"],
                pred["tta_passes"],
            )
            results.append({
                "id":             row_id,
                "filename":       filename,
                "status":         "Analyzed",
                "prediction":     pred["prediction"],
                "confidence":     pred["confidence"],
                "severity_score": pred["severity_score"],
                "normal_prob":    pred["normal_prob"],
                "pneumonia_prob": pred["pneumonia_prob"],
                "tta_passes":     pred["tta_passes"],
                "priority":       _priority_label(pred["severity_score"]),
            })
        except Exception as exc:
            errors.append({"filename": filename, "error": str(exc)})

    results.sort(key=lambda x: x["severity_score"], reverse=True)

    return jsonify({
        "processed":     len(results),
        "errors":        len(errors),
        "results":       results,
        "error_details": errors,
    })


@app.route("/api/triage/queue", methods=["GET"])
def triage_queue():
    """
    Return all reports sorted by severity_score descending.
    Query params: ?status=Pending|Analyzed|all  ?limit=N
    """
    ensure_db()

    status = request.args.get("status", "all")
    limit  = int(request.args.get("limit", 100))

    conn = _get_db()
    if status == "all":
        rows = conn.execute(
            "SELECT * FROM reports ORDER BY severity_score DESC NULLS LAST LIMIT ?",
            (limit,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM reports WHERE status=? "
            "ORDER BY severity_score DESC NULLS LAST LIMIT ?",
            (status, limit),
        ).fetchall()
    conn.close()

    queue = []
    for r in rows:
        sev = r["severity_score"]
        queue.append({
            "id":             r["id"],
            "filename":       r["filename"],
            "status":         r["status"],
            "prediction":     r["prediction"],
            "confidence":     round(r["confidence"], 2) if r["confidence"] is not None else None,
            "severity_score": round(sev, 2)             if sev             is not None else None,
            "tta_passes":     r["tta_passes"],
            "priority":       _priority_label(sev),
            "timestamp":      r["timestamp"],
        })

    return jsonify({"total": len(queue), "queue": queue})


@app.route("/api/triage/clear", methods=["DELETE"])
def triage_clear():
    """Delete all triage records — for development and testing."""
    ensure_db()
    conn = _get_db()
    conn.execute("DELETE FROM reports")
    conn.commit()
    conn.close()
    return jsonify({"message": "Triage queue cleared."})


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ensure_db()
    load_pneumonia_model()   # pre-warm — surface missing checkpoint immediately
    print("[INFO] Starting on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)