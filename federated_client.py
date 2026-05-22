"""
federated_client.py — Federated Learning Hospital Node
=======================================================
A self-contained Flask server representing a single hospital participant.
It holds an isolated data partition and performs one local training epoch
whenever the central server dispatches global weights via HTTP.

STRICT DATA ISOLATION: This node only reads images from DATA_DIR as supplied
on the command line. It never accesses any other directory.

Usage:
    python federated_client.py <PORT> <DATA_DIR>

    Example (3 terminals):
        python federated_client.py 5001 data/hospital_A
        python federated_client.py 5002 data/hospital_B
        python federated_client.py 5003 data/hospital_C

Endpoint:
    POST /train_round
        Input  : multipart/form-data, field 'weights' — serialized state_dict bytes
        Output : application/octet-stream — updated state_dict bytes (200 OK)
        Errors : 400 Bad Request | 500 Internal Server Error
"""

import io
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from torch.utils.data import DataLoader
from torchvision import datasets, models

# ── Argument validation ───────────────────────────────────────────────────────
if len(sys.argv) < 3:
    print("Usage: python federated_client.py <PORT> <DATA_DIR>", file=sys.stderr)
    sys.exit(1)

try:
    PORT = int(sys.argv[1])
except ValueError:
    print(f"Error: PORT must be an integer, got '{sys.argv[1]}'", file=sys.stderr)
    sys.exit(1)

DATA_DIR = Path(sys.argv[2]).resolve()
if not DATA_DIR.exists():
    print(f"Error: DATA_DIR '{DATA_DIR}' does not exist.", file=sys.stderr)
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────
BATCH_SIZE    = 16
LEARNING_RATE = 0.0005
IMAGE_SIZE    = 224
NUM_CLASSES   = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Flask setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Data loading ─────────────────────────────────────────────────────────────
# The transform is shared across all requests since it never changes.
_train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _build_dataloader() -> DataLoader:
    """
    Construct a DataLoader from DATA_DIR.  The directory must contain
    exactly the subdirectory layout produced by data_splitter.py:
        DATA_DIR/NORMAL/
        DATA_DIR/PNEUMONIA/
    """
    dataset = datasets.ImageFolder(str(DATA_DIR), transform=_train_transform)
    if not dataset.samples:
        raise RuntimeError(
            f"No images found under '{DATA_DIR}'. "
            "Run data_splitter.py first."
        )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


def _build_model() -> nn.Module:
    """
    Instantiate EfficientNet-B0 with the regularised classifier head.

    This head MUST be identical to the one used in simple_backend.py and
    train_model.py — all three files must stay in sync or state_dict keys
    will mismatch and loading will crash with 'Unexpected key(s)' errors.

    Head layout:
        Dropout(0.4) → Linear(1280, 512) → ReLU →
        BatchNorm1d(512) → Dropout(0.2) → Linear(512, 2)
    """
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features   # 1280 for EfficientNet-B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.2),
        nn.Linear(512, NUM_CLASSES),
    )
    return model


def _deserialize_state_dict(weight_bytes: bytes) -> dict:
    """Load a PyTorch state_dict from raw bytes."""
    buf = io.BytesIO(weight_bytes)
    # weights_only=True is safe here: we only serialise state_dicts (tensor dicts).
    return torch.load(buf, map_location=device, weights_only=True)


def _serialize_state_dict(state_dict: dict) -> bytes:
    """Serialise a PyTorch state_dict to raw bytes."""
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)
    return buf.read()


def _train_one_epoch(model: nn.Module, loader: DataLoader) -> float:
    """
    Execute exactly one local training epoch.
    Returns the average loss for logging/debugging.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_loss  = 0.0
    num_batches = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss  += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/train_round", methods=["POST"])
def train_round():
    """
    Federated training round endpoint.

    1. Receive the global weight bytes from the central server.
    2. Load them into a fresh EfficientNet-B0 instance.
    3. Run one local training epoch on DATA_DIR images.
    4. Return the updated state_dict as a binary stream.
    """
    if "weights" not in request.files:
        return jsonify({"error": "Missing 'weights' field in multipart request."}), 400

    weight_file = request.files["weights"]

    try:
        weight_bytes = weight_file.read()
        if not weight_bytes:
            return jsonify({"error": "Received empty weights payload."}), 400

        # 1. Deserialise global weights → model
        global_state_dict = _deserialize_state_dict(weight_bytes)
        model = _build_model()
        model.load_state_dict(global_state_dict)
        model.to(device)

        # 2. Build the local DataLoader (locked to DATA_DIR)
        loader = _build_dataloader()
        num_samples = len(loader.dataset)

        # 3. Local training epoch
        avg_loss = _train_one_epoch(model, loader)
        print(f"[Node port={PORT}] Round complete — "
              f"samples={num_samples}, avg_loss={avg_loss:.4f}")

        # 4. Serialise and stream back updated weights
        updated_bytes = _serialize_state_dict(model.state_dict())
        return Response(
            updated_bytes,
            status=200,
            mimetype="application/octet-stream",
        )

    except RuntimeError as exc:
        print(f"[Node port={PORT}] RuntimeError: {exc}")
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Simple liveness probe for the orchestrator."""
    try:
        loader     = _build_dataloader()
        num_images = len(loader.dataset)
    except Exception:
        num_images = -1
    return jsonify({
        "status":    "ok",
        "port":      PORT,
        "data_dir":  str(DATA_DIR),
        "num_images": num_images,
        "device":    str(device),
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[INFO] Hospital node starting on port {PORT}")
    print(f"[INFO] Data directory : {DATA_DIR}")
    print(f"[INFO] Device         : {device}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
