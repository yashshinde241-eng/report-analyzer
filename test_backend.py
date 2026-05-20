"""
test_backend.py — Report Analyzer API Integration Tests
=========================================================
Tests the live Flask backend endpoints.  Run this script *after* starting
the server:

    python simple_backend.py

Usage:
    python test_backend.py [--url http://localhost:5000]
"""

import argparse
import io
import json
import sys

import requests
from PIL import Image

# ── CLI argument for flexible target URL ─────────────────────────────────────
parser = argparse.ArgumentParser(description="Report Analyzer backend tests")
parser.add_argument(
    "--url",
    default="http://localhost:5000",
    help="Base URL of the running backend (default: http://localhost:5000)",
)
args, _ = parser.parse_known_args()
BACKEND_URL = args.url.rstrip("/")

PASS  = "✅ PASS"
FAIL  = "❌ FAIL"
SKIP  = "⚠️  SKIP"
SEP   = "=" * 60


def _section(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def _result(label: str, ok: bool | None, detail: str = "") -> bool | None:
    icon = PASS if ok is True else (SKIP if ok is None else FAIL)
    print(f"{icon} {label}")
    if detail:
        print(f"    {detail}")
    return ok


# ── Test 1: Health Check ──────────────────────────────────────────────────────

def test_health_check() -> bool:
    _section("TEST 1: Health Check — GET /api/health")
    try:
        resp = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
        ok   = resp.status_code == 200
        data = resp.json()
        _result("HTTP 200 OK", ok)
        _result("status == 'ok'",  data.get("status") == "ok",
                f"status={data.get('status')}")
        _result("'device' key present",    "device"            in data)
        _result("'groq_configured' key",   "groq_configured"   in data)
        _result("'model_checkpoint' key",  "model_checkpoint"  in data,
                f"checkpoint found = {data.get('model_checkpoint')}")
        return ok
    except requests.exceptions.ConnectionError:
        _result("Connection", False,
                f"Cannot reach backend at {BACKEND_URL}. Is it running?")
        return False
    except Exception as exc:
        _result("Unexpected error", False, str(exc))
        return False


# ── Test 2: Single Image Analysis ─────────────────────────────────────────────

def test_single_image_analysis() -> bool:
    _section("TEST 2: Single Image — POST /api/analyze/image  (field: 'file')")
    try:
        # Build a minimal valid RGB PNG in memory
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), color=(128, 128, 128)).save(buf, format="PNG")
        buf.seek(0)

        resp = requests.post(
            f"{BACKEND_URL}/api/analyze/image",
            files={"file": ("test_xray.png", buf, "image/png")},
            timeout=30,
        )
        data = resp.json()
        ok   = resp.status_code == 200
        _result("HTTP 200 OK", ok, f"status={resp.status_code}")
        _result("'prediction' in response",     "prediction"    in data)
        _result("'confidence' in response",     "confidence"    in data)
        _result("'severity_score' in response", "severity_score" in data)
        _result("'detected' in response",       "detected"       in data)
        _result("'disease' == 'Pneumonia'",
                data.get("disease") == "Pneumonia", f"disease={data.get('disease')}")
        # Validate 0–100 scale
        conf = data.get("confidence")
        sev  = data.get("severity_score")
        _result("confidence in [0,100]",
                conf is not None and 0.0 <= conf <= 100.0,
                f"confidence={conf}")
        _result("severity_score in [0,100]",
                sev is not None  and 0.0 <= sev  <= 100.0,
                f"severity_score={sev}")
        return ok
    except Exception as exc:
        _result("Unexpected error", False, str(exc))
        return False


# ── Test 3: Triage Analyze-One ────────────────────────────────────────────────

def test_triage_analyze_one() -> bool:
    _section("TEST 3: Triage Single — POST /api/triage/analyze-one")
    try:
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), color=(60, 60, 60)).save(buf, format="JPEG")
        buf.seek(0)

        resp = requests.post(
            f"{BACKEND_URL}/api/triage/analyze-one",
            files={"file": ("patient_001.jpg", buf, "image/jpeg")},
            timeout=30,
        )
        data = resp.json()
        ok   = resp.status_code == 200
        _result("HTTP 200 OK", ok, f"status={resp.status_code}")
        _result("'id' in response",           "id"             in data)
        _result("'priority' in response",     "priority"       in data,
                f"priority={data.get('priority')}")
        _result("status == 'Analyzed'",
                data.get("status") == "Analyzed", f"status={data.get('status')}")
        sev = data.get("severity_score")
        _result("severity_score in [0,100]",
                sev is not None and 0.0 <= sev <= 100.0, f"severity_score={sev}")
        valid_priorities = {"High", "Medium", "Low"}
        _result("priority is valid label",
                data.get("priority") in valid_priorities,
                f"priority={data.get('priority')}")
        return ok
    except Exception as exc:
        _result("Unexpected error", False, str(exc))
        return False


# ── Test 4: Triage Queue ──────────────────────────────────────────────────────

def test_triage_queue() -> bool:
    _section("TEST 4: Triage Queue — GET /api/triage/queue")
    try:
        resp = requests.get(f"{BACKEND_URL}/api/triage/queue", timeout=10)
        data = resp.json()
        ok   = resp.status_code == 200
        _result("HTTP 200 OK", ok, f"status={resp.status_code}")
        _result("'total' in response",  "total" in data)
        _result("'queue' is a list",    isinstance(data.get("queue"), list))

        queue = data.get("queue", [])
        if queue:
            first = queue[0]
            _result("Queue item has 'severity_score'",
                    "severity_score" in first)
            _result("Queue item has 'priority'",
                    "priority" in first)
            # Verify descending severity order
            sevs = [r["severity_score"] for r in queue
                    if r["severity_score"] is not None]
            ordered = sevs == sorted(sevs, reverse=True)
            _result("Queue sorted by severity DESC", ordered,
                    f"scores={sevs[:5]}{'...' if len(sevs)>5 else ''}")
        else:
            _result("Queue ordering (skipped — empty queue)", None)
        return ok
    except Exception as exc:
        _result("Unexpected error", False, str(exc))
        return False


# ── Test 5: MIME Validation ───────────────────────────────────────────────────

def test_mime_validation() -> bool:
    _section("TEST 5: Input Validation — reject non-image uploads")
    try:
        # Send a text file disguised as an upload
        resp = requests.post(
            f"{BACKEND_URL}/api/analyze/image",
            files={"file": ("report.txt", b"this is not an image", "text/plain")},
            timeout=10,
        )
        ok = resp.status_code == 400
        _result("HTTP 400 for .txt file", ok,
                f"status={resp.status_code}")
        data = resp.json()
        _result("Error message present", bool(data.get("error")),
                f"error={data.get('error')}")
        return ok
    except Exception as exc:
        _result("Unexpected error", False, str(exc))
        return False


# ── Test 6: Triage Clear ──────────────────────────────────────────────────────

def test_triage_clear() -> bool:
    _section("TEST 6: Triage Clear — DELETE /api/triage/clear")
    try:
        resp = requests.delete(f"{BACKEND_URL}/api/triage/clear", timeout=10)
        ok   = resp.status_code == 200
        _result("HTTP 200 OK", ok, f"status={resp.status_code}")
        data = resp.json()
        _result("'message' in response", "message" in data,
                f"message={data.get('message')}")
        return ok
    except Exception as exc:
        _result("Unexpected error", False, str(exc))
        return False


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all_tests() -> None:
    print(f"\n{SEP}")
    print("REPORT ANALYZER — BACKEND INTEGRATION TESTS")
    print(f"Target: {BACKEND_URL}")
    print(SEP)

    # Health check gates all other tests
    if not test_health_check():
        print(f"\n{SEP}")
        print("ABORTED — backend unreachable. Start simple_backend.py first.")
        print(SEP)
        sys.exit(1)

    results = {
        "Health Check":        True,  # already passed above
        "Single Image":        test_single_image_analysis(),
        "Triage Analyze-One":  test_triage_analyze_one(),
        "Triage Queue":        test_triage_queue(),
        "MIME Validation":     test_mime_validation(),
        "Triage Clear":        test_triage_clear(),
    }

    _section("SUMMARY")
    passed  = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is None)
    failed  = sum(1 for v in results.values() if v is False)

    for name, res in results.items():
        icon = PASS if res is True else (SKIP if res is None else FAIL)
        print(f"{icon}  {name}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("\n🎉 All tests passed. Backend is healthy and ready.")
    else:
        print("\n⚠️  Some tests failed — review output above.")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
