# SESSION_STATE.md — Report Analyzer Phase 1
**Last Updated:** Phase 1 complete  
**Status:** All Section A refactoring done · All Section B FL files built · Ready for Phase 2

---

## 1. Executive Summary

`report-analyzer` is a Flask-based medical imaging web application that classifies chest X-rays for pneumonia using an EfficientNet-B0 PyTorch model, generates structured AI reasoning via the Groq Llama 3.1 API, and provides a priority-sorted triage queue backed by SQLite.

**Phase 1** delivered two major upgrades:

- **Section A — Stabilisation:** Hardcoded Windows paths eliminated, severity scores normalised to a uniform 0–100 scale across all DB writes and API responses, MIME/extension validation added to all upload endpoints, PyTorch loading modernised with `weights_only=True`, a graceful startup model-missing guard implemented, and `test_backend.py` fully rewritten to match the actual endpoint contracts.

- **Section B — Federated Learning:** Three new files (`data_splitter.py`, `federated_client.py`, `federated_server.py`) implement a fully operational simulated multi-hospital Federated Averaging (FedAvg) pipeline running over localhost ports 5001–5003. Orphaned diabetes/text-classification code (`train_text_model.py`, `generate_text_data.py`) is deprecated and can be deleted.

---

## 2. Section A — Completed Refactoring Details

### 2.1 Path Portability
| File | Change |
|---|---|
| `simple_backend.py` | All paths replaced with `Path(__file__).resolve().parent / ...` |
| `train_model.py` | Dataset and model paths now CLI args with `argparse`; default auto-detects relative `data set/chest_xray` |
| `test_model.py` | Model and image paths now CLI args; no hardcoded paths remain |
| `data_splitter.py` | Source and output paths are all relative to project root |

### 2.2 Severity Score Standardisation
- `predict_pneumonia()` now converts to `0–100` **immediately** before returning:  
  `severity_score_100 = round(raw_severity * 100, 2)`  
  `confidence_100 = round(confidence * 100, 2)`
- `update_analyzed()` receives and stores `0–100` values; the DB column now holds the same scale.
- All three triage endpoints (`analyze-one`, `upload-bulk`, `queue`) read from and write to this uniform scale.
- Priority thresholds: `severity >= 70` → High, `>= 40` → Medium, `< 40` → Low — evaluated correctly everywhere via `_priority_label()`.

### 2.3 MIME / Extension Validation
- New helper `_validate_image_file(file)` checks extension ∈ `{png, jpg, jpeg}` and MIME type ∈ `{image/png, image/jpeg}`.
- Applied to: `/api/analyze/image`, `/api/triage/analyze-one`, `/api/triage/upload-bulk`.
- Invalid files return `400 Bad Request` JSON `{"error": "..."}` — no raw stack traces exposed.

### 2.4 PyTorch Safe Loading
- All `torch.load(...)` calls now include `weights_only=True`:
  ```python
  torch.load(str(MODEL_PATH), map_location=device, weights_only=True)
  ```
  Applied in `simple_backend.py`, `test_model.py`, `federated_client.py`, and `federated_server.py`.

### 2.5 Graceful Model Startup Guard
In `simple_backend.py`, `load_pneumonia_model()` checks `PNEUMONIA_MODEL_PATH.exists()`.  
If missing: prints a loud bordered warning, initialises with random weights, continues running.  
Server never crashes on startup due to a missing model file.  
The `/api/health` endpoint exposes `"model_checkpoint": true/false` so you can verify remotely.

### 2.6 test_backend.py Rewrite
- Removed: all references to `/api/analyze/file` and text/diabetes uploads.
- Fixed: image test now uses field `'file'` and endpoint `/api/analyze/image`.
- Added: 6 targeted tests — health check, single image (with 0–100 scale validation), triage analyze-one, triage queue (with sort order assertion), MIME validation rejection, and triage clear.
- Added: `--url` CLI flag for flexible target overriding.
- Health check gates all other tests; exits with code 1 on any failure.

### 2.7 Deprecated Files
The following files are orphaned and should be deleted at your convenience.  
They are not referenced by any active code:
- `train_text_model.py`
- `generate_text_data.py`

### 2.8 requirements.txt
- All ML packages uncommented and pinned: `flask`, `flask-cors`, `werkzeug`, `requests`, `python-dotenv`, `pillow`, `torch`, `torchvision`, `numpy`.
- Removed: `scikit-learn`, `joblib`, `PyPDF2`, `python-docx`, `fastapi`, `uvicorn`.

---

## 3. Section B — Federated Learning Infrastructure

### 3.1 Architecture Overview

```
  ┌─────────────────────────────────────────────────────┐
  │              federated_server.py                    │
  │  (Orchestrator — no raw data access)                │
  │  - Initialises global EfficientNet-B0               │
  │  - Dispatches global weights → 3 nodes              │
  │  - Collects updated weights from each node          │
  │  - Applies FedAvg: θ_global ← (1/N)Σθ_client_i     │
  │  - Saves checkpoint → models/pneumonia_model.pth    │
  └────────────┬───────────────┬───────────────┬────────┘
               │               │               │
               ▼               ▼               ▼
  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
  │ federated_     │  │ federated_     │  │ federated_     │
  │ client.py      │  │ client.py      │  │ client.py      │
  │ port=5001      │  │ port=5002      │  │ port=5003      │
  │ data/hospital_A│  │ data/hospital_B│  │ data/hospital_C│
  └────────────────┘  └────────────────┘  └────────────────┘
```

### 3.2 data_splitter.py
| Property | Detail |
|---|---|
| Input | `data set/chest_xray/train/` (Kaggle dataset) |
| Output | `data/hospital_A/`, `data/hospital_B/`, `data/hospital_C/` |
| Strategy | Even round-robin split per class after random shuffle (seed=42) |
| Class preservation | NORMAL and PNEUMONIA subdirs replicated inside each hospital dir |
| Idempotent | Safe to re-run; existing files are overwritten by `shutil.copy2` |
| CLI arg | `--dataset /custom/path --seed 42` |

### 3.3 federated_client.py
| Property | Detail |
|---|---|
| Launch | `python federated_client.py <PORT> <DATA_DIR>` |
| Data isolation | Strictly reads only `DATA_DIR`; no cross-node data access possible |
| Model | Fresh EfficientNet-B0 (no weights), classifier head replaced for 2-class output |
| Training | 1 local epoch, Adam optimizer, lr=0.0005, CrossEntropyLoss |
| Input | `POST /train_round` — multipart field `weights` (binary state_dict) |
| Output | `200 OK`, `application/octet-stream` — updated binary state_dict |
| Error handling | Returns `400` for missing/empty payload; `500` with JSON error for runtime failures |
| Bonus endpoint | `GET /health` — returns port, data_dir, num_images, device |

### 3.4 federated_server.py
| Property | Detail |
|---|---|
| Registered nodes | `http://127.0.0.1:5001`, `5002`, `5003` |
| Global model | EfficientNet-B0, IMAGENET1K_V1 ImageNet pre-weights |
| Pre-round probe | `GET /health` on each node; unreachable nodes are skipped (not fatal) |
| Communication | Sequential HTTP POST to each alive node; configurable `--timeout` (default 300s) |
| Aggregation | FedAvg: per-layer tensor stack + `mean(dim=0)` across all responding clients |
| Checkpoint | Saved after every round with keys: `round`, `model_state_dict`, `num_clients`, `classes` |
| CLI args | `--rounds N`, `--output path`, `--timeout seconds` |

---

## 4. Current Execution State

### File Checklist
| File | Status |
|---|---|
| `simple_backend.py` | ✅ Refactored & complete |
| `train_model.py` | ✅ Paths portable, `weights_only` updated |
| `test_model.py` | ✅ Paths portable, `weights_only` updated, CLI args |
| `test_backend.py` | ✅ Fully rewritten |
| `requirements.txt` | ✅ Clean, all ML deps uncommented |
| `init_db.py` | ✅ Unchanged (already correct) |
| `data_splitter.py` | ✅ New — built Phase 1 |
| `federated_client.py` | ✅ New — built Phase 1 |
| `federated_server.py` | ✅ New — built Phase 1 |
| `train_text_model.py` | ⚠️ Deprecated — safe to delete |
| `generate_text_data.py` | ⚠️ Deprecated — safe to delete |

### How to Run the Full System (4 Terminals)

**Terminal 1 — Main API backend:**
```bash
python simple_backend.py
# Runs on http://localhost:5000
```

**Terminals 2–4 — Federated hospital nodes (after running data_splitter.py):**
```bash
# First, split the data:
python data_splitter.py

# Then start each node in a separate terminal:
python federated_client.py 5001 data/hospital_A
python federated_client.py 5002 data/hospital_B
python federated_client.py 5003 data/hospital_C
```

**Federated training round (separate terminal):**
```bash
python federated_server.py --rounds 5
# Aggregated model saved to models/pneumonia_model.pth
```

**Run tests:**
```bash
# Backend must be running first
python test_backend.py

# Single model test (checkpoint must exist):
python test_model.py --image /path/to/xray.jpg
```

### Known Issues Resolved
- ✅ `torch.load` deprecation warnings (fixed with `weights_only=True`)
- ✅ Severity score inconsistency between `analyze-one` (0–100) and `queue` (0–1) — now uniform 0–100 everywhere
- ✅ `test_backend.py` testing non-existent `/api/analyze/file` endpoint — rewritten
- ✅ Windows absolute paths crashing on non-Windows machines — eliminated
- ✅ Server crash on missing model file — replaced with graceful warning + random weights

---

## 5. Phase 2 Hand-off Protocol

### What is Complete and Ready
- The main Flask backend (`simple_backend.py`) exposes all endpoints the frontend needs; the `/api/triage/analyze-one` endpoint is already designed for per-file progressive updates (one HTTP call per image).
- All severity/confidence values returned are on a consistent 0–100 scale, ready for display in progress bars.
- The Federated Learning pipeline is fully wired end-to-end on localhost; all 3 FL scripts are production-ready.
- The Groq integration function `get_ai_reasoning()` exists and works; it is currently called only in `/api/analyze/image`.

### Phase 2 Targets
1. **Frontend SSE Integration:** Add a `GET /api/triage/stream` Server-Sent Events endpoint in `simple_backend.py` that emits per-image progress events during bulk uploads. Update `report-analyzer.html` to consume the event stream and render a real-time progress bar.

2. **Real-Time Progress Bars:** The frontend's `submitBulk()` currently fires N parallel `fetch` calls to `analyze-one`. Migrate this to the SSE stream endpoint so progress is server-driven and doesn't depend on client-side Promise.all timing.

3. **Privacy-Preserving Groq Pipeline:** Extend the Groq reasoning call to accept an optional `noise_sigma` parameter and add differential-privacy noise to the confidence value before sending it to the external Groq API. This ensures the raw model confidence is never transmitted in plain form to a third-party service.

4. **Federated Training Trigger Endpoint:** Add a `POST /api/federated/start-round` endpoint to `simple_backend.py` that programmatically launches a federated round (calling `federated_server.py` logic) and streams round-by-round progress back to the frontend via SSE.
