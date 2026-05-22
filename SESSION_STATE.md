# SESSION_STATE.md — Report Analyzer Phase 1 + 1.5 + 2
**Last Updated:** Phase 2 complete
**Status:** All sections built, verified, and ready for use

---

## 1. Executive Summary

`report-analyzer` is a full-stack medical imaging web application comprising:

- **Flask backend** (`simple_backend.py`) — REST API + SSE streaming, local EfficientNet-B0 TTA inference, privacy-preserving Groq pipeline, SQLite triage queue
- **Vanilla JS frontend** (`report-analyzer.html`) — 3-panel SPA: Analysis, Triage Dashboard, Federated Control Tower
- **Federated Learning pipeline** — 3 simulated hospital nodes (`federated_client.py`) coordinated by a FedAvg orchestrator (`federated_server.py`)
- **Robustness layer** (`xray_transforms.py`) — CLAHE preprocessing, Albumentations augmentation, TTA inference

**Data privacy guarantee:** Zero images or pixel data leave the local server. Only anonymised numeric text is sent to the external Groq API.

---

## 2. Phase 1 — Codebase Stabilisation

### 2.1 Changes Made
| File | Change |
|---|---|
| `simple_backend.py` | Portable paths, 0–100 severity scale everywhere, MIME validation, startup model guard, `weights_only=True` |
| `train_model.py` | Argparse CLI, portable paths, `weights_only=True` |
| `test_model.py` | Argparse CLI, portable paths, `weights_only=True` |
| `test_backend.py` | Fully rewritten — 6 tests, correct endpoints and field names |
| `requirements.txt` | All ML deps uncommented, diabetes deps removed |

### 2.2 Severity Scale
All metrics are stored and returned on a uniform **0–100 float scale** throughout:
- `predict_pneumonia()` converts raw 0–1 probabilities immediately
- `update_analyzed()` stores 0–100 values in SQLite
- All three triage endpoints read, write, and return 0–100
- Priority thresholds: `>= 70` → High, `>= 40` → Medium, `< 40` → Low

### 2.3 Deprecated Files
Safe to delete — not referenced by any active code:
- `train_text_model.py`
- `generate_text_data.py`

---

## 3. Phase 1.5 — Robustness Techniques

New file: **`xray_transforms.py`** — single source of truth for all image transforms.

| Technique | Implementation | Where Used |
|---|---|---|
| CLAHE preprocessing | `apply_clahe()` — grayscale → CLAHE → RGB | Training dataset, all inference |
| Albumentations pipeline | `TRAIN_TRANSFORM`, `VAL_TRANSFORM` | `train_model.py` |
| Label smoothing | `CrossEntropyLoss(label_smoothing=0.1)` | `train_model.py` |
| Cosine annealing WR | `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)` | `train_model.py` |
| Mixup (α=0.4) | `mixup_data()` + `mixup_criterion()` | `train_model.py` |
| TTA inference | `run_tta_inference()` — 4 or 8 augmented passes | `simple_backend.py` (all endpoints) |

**TTA level:** Controlled via `TTA_LEVEL` env var (`fast`=4 passes, `full`=8). Default: `fast`.

### 3.1 Augmentation Detail (TRAIN_TRANSFORM)
- ElasticTransform (α=120, σ=6) — simulates breathing artefacts
- GridDistortion — scanner geometric distortion
- RandomBrightnessContrast (±30%) — scanner calibration variance
- GaussNoise (σ=0.02–0.12) — electronic noise at low dose
- CLAHE (p=0.4) — randomly applied so model handles raw and processed inputs
- CoarseDropout (1–6 holes, 8–24px) — foreign objects, occlusions
- HorizontalFlip, RandomRotate90

---

## 4. Phase 2 — Live UI & Privacy Pipeline

### 4.1 Privacy-Preserving Groq Pipeline

**`/api/analyze/image` data flow (Task 3):**

```
[Raw image bytes] ──► local EfficientNet-B0 TTA ──► numeric metrics
                                                         │
                                                         ▼
                                          anonymised text string
                                     "LOCAL_ANALYSIS: Class=..., Confidence=...%,
                                      Triage_Severity=.../100, P(Normal)=...%,
                                      P(Pneumonia)=...%, Method=EfficientNet-B0_TTA_4pass"
                                                         │
                                                         ▼
                                              Groq API (text only)
                                              Llama-3.1-8b-instant
                                                         │
                                                         ▼
                                          clinical reasoning text
                                                         │
                                    ┌────────────────────┘
                                    ▼
                         combined JSON response to frontend
                         {prediction, confidence, severity_score,
                          normal_prob, pneumonia_prob, tta_passes,
                          privacy: {image_sent_externally: false,
                                    anonymised_summary: "...",
                                    external_service: "Groq/..."},
                          reasoning: "..."}
```

**Privacy guarantees:**
- `image_sent_externally: false` — always; raw bytes never leave the Flask process
- Groq prompt explicitly instructs Llama not to infer an image exists
- Groq prompt explicitly instructs Llama not to hallucinate patient details
- The `anonymised_summary` string is surfaced in the frontend as a transparency badge

### 4.2 SSE Architecture (Task 1)

**Endpoint:** `GET /api/federated/stream` → `text/event-stream`

**Internal mechanism:**
- `_sse_queue: queue.Queue` — thread-safe FIFO, max 512 events
- `_push_event(type, payload)` — non-blocking push from FL thread
- SSE generator drains queue with 25s timeout; emits `: keepalive` on empty

**Event types emitted:**

| Event type | Payload fields | Trigger |
|---|---|---|
| `connected` | `message` | On EventSource open |
| `round_start` | `round`, `total` | Each round begins |
| `node_status` | `node`, `id`, `status` | Node goes training/idle/error/offline |
| `log_stream` | `log` | Every major orchestrator action |
| `round_complete` | `round`, `global_accuracy`, `num_clients` | After FedAvg applied |
| `fl_done` | `rounds` | All rounds finished |
| `fl_error` | `message` | No nodes reachable |

**Threading:** `app.run(threaded=True)` — essential. The SSE generator blocks in its thread while the FL orchestrator runs in a `daemon=True` background thread. Both coexist safely via the queue.

### 4.3 Federated Control Tower UI (Task 2)

**New tab:** "🛰 Fed. Tower" — `panel-fl` section in `report-analyzer.html`

**Components:**

**Node monitors (3 cards, one per hospital):**
- Status badge: Idle / 🟢 Training / Offline / ⚠ Error
- Circular SVG ring with CSS `stroke-dasharray`/`stroke-dashoffset` animation
- `ringPulse` keyframe: animates ring fill 220→55→220 dashoffset while training
- Card glows cyan (`box-shadow: 0 0 24px rgba(0,212,255,.12)`) during training
- Top border sweep animation (gradient bar) activates on training state

**HUD Terminal console:**
- Dark `#080810` background, JetBrains Mono, cyan text
- macOS-style dot header (`terminal-header`)
- Scrolling `terminal-body` div, auto-scrolls to bottom on each log line
- Blinking cursor appended to last line
- Colour coding: green (normal), red (error/fail), dim (metadata)
- Pruned to 200 lines max

**Live accuracy chart (Chart.js 4.4.1 via CDN):**
- Line chart, `fill: true`, cyan gradient
- X axis: `R1`, `R2`, … (round labels)
- Y axis: 50–100% accuracy range
- Dark theme: `#10101a` tooltip background, cyan border
- `pushChartPoint(round, accuracy)` called on each `round_complete` event
- "↺ Reset" button clears chart data

**Control panel:**
- "⚡ Trigger Training Round" — cyan glow button, disabled during active round
- Rounds input (1–10, clamped server-side)
- SSE connection dot indicator (grey → green pulsing → red on error)

**JS EventSource client:**
- `connectSSE()` — creates `EventSource`, wires `onmessage`, `onopen`, `onerror`
- `_sseSource` module-level singleton, reconnectable
- `initFLPanel()` — called once on first tab open; inits chart + SSE + node probe
- Node probe: `GET /api/federated/nodes` on tab open to set initial online/offline state

### 4.4 New Endpoints Added

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/federated/stream` | SSE event stream |
| `POST` | `/api/federated/trigger` | Start FL round(s) in background thread |
| `GET` | `/api/federated/nodes` | Live health status of all 3 nodes |

---

## 5. Current Execution State

### File Checklist
| File | Status |
|---|---|
| `simple_backend.py` | ✅ Phase 2 complete — SSE, privacy pipeline, FL trigger |
| `report-analyzer.html` | ✅ Phase 2 complete — 3 panels, Control Tower, SSE client |
| `xray_transforms.py` | ✅ Phase 1.5 complete — CLAHE, Albumentations, TTA |
| `train_model.py` | ✅ Phase 1.5 complete — all robustness techniques |
| `test_model.py` | ✅ Phase 1 complete — portable CLI |
| `test_backend.py` | ✅ Phase 1 complete — 6 tests |
| `requirements.txt` | ✅ Clean — all deps uncommented |
| `data_splitter.py` | ✅ Phase 1 complete |
| `federated_client.py` | ✅ Phase 1 complete |
| `federated_server.py` | ✅ Phase 1 complete (standalone CLI) |
| `SESSION_STATE.md` | ✅ Phase 2 updated |
| `train_text_model.py` | ⚠️ Deprecated — safe to delete |
| `generate_text_data.py` | ⚠️ Deprecated — safe to delete |

### How to Run (4+ Terminals)

**Terminal 1 — Main backend:**
```bash
python simple_backend.py
# http://localhost:5000
```

**Terminals 2–4 — FL hospital nodes (optional, only needed for FL tab):**
```bash
python data_splitter.py                          # run once
python federated_client.py 5001 data/hospital_A
python federated_client.py 5002 data/hospital_B
python federated_client.py 5003 data/hospital_C
```

**Open the frontend:**
```
Open report-analyzer.html in a browser
```

**Run tests (backend must be running):**
```bash
python test_backend.py
```

---

## 6. Phase 3 Hand-off Protocol

### Ready to integrate
- SSE infrastructure is live and proven; adding new event types requires only `_push_event()` calls
- The `privacy` field in `/api/analyze/image` response is structured for easy extension
- Chart.js is already loaded and `pushChartPoint()` is exposed globally
- All node state management is centralised in `setNodeStatus(nodeId, status)`

### Phase 3 candidates
1. **Real accuracy evaluation** — wire a held-out validation set into `_run_fl_round_thread` and replace mock accuracy with real per-round test metrics
2. **Differential privacy noise** — add Gaussian noise to confidence before Groq call: `conf_noised = conf + N(0, σ²)` with configurable σ
3. **Multi-disease support** — extend the model head to N classes; update `_build_anonymised_summary` and frontend badge rendering
4. **Auth layer** — add JWT token validation to `/api/federated/trigger` to prevent unauthorised FL round initiation
5. **Docker Compose** — containerise all 5 processes (backend + 3 nodes + nginx) for one-command deployment
