"""
federated_server.py — Federated Learning Orchestrator (Central Server)
=======================================================================
Coordinates the FedAvg algorithm across 3 hospital nodes.  This server
never touches raw patient data; it only exchanges model weight tensors.

The FedAvg update rule applied after each round:

    θ_global ← (1/N) * Σ θ_client_i

Usage:
    # 1. Ensure all 3 client nodes are running:
    #       python federated_client.py 5001 data/hospital_A
    #       python federated_client.py 5002 data/hospital_B
    #       python federated_client.py 5003 data/hospital_C
    #
    # 2. Then run this orchestrator:
    python federated_server.py [--rounds 5] [--output models/pneumonia_model.pth]
"""

import argparse
import io
from pathlib import Path

import requests
import torch
import torch.nn as nn
from torchvision import models

# ── CLI ───────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent
DEFAULT_OUT  = ROOT_DIR / "models" / "pneumonia_model.pth"

parser = argparse.ArgumentParser(description="Federated Learning Orchestrator")
parser.add_argument(
    "--rounds",
    type=int,
    default=3,
    help="Number of federated communication rounds (default: 3).",
)
parser.add_argument(
    "--output",
    default=str(DEFAULT_OUT),
    help="Path to save the final aggregated model checkpoint.",
)
parser.add_argument(
    "--timeout",
    type=int,
    default=300,
    help="HTTP timeout (seconds) per client request (default: 300).",
)
args = parser.parse_args()

NUM_ROUNDS  = args.rounds
OUTPUT_PATH = Path(args.output)
TIMEOUT     = args.timeout

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Client node registry ──────────────────────────────────────────────────────
CLIENT_NODES = [
    "http://127.0.0.1:5001",
    "http://127.0.0.1:5002",
    "http://127.0.0.1:5003",
]

# ── Model factory ─────────────────────────────────────────────────────────────
NUM_CLASSES = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_global_model() -> nn.Module:
    """Instantiate a fresh EfficientNet-B0 with ImageNet pre-weights."""
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.to(device)
    model.eval()
    return model


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _serialize_state_dict(state_dict: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)
    return buf.read()


def _deserialize_state_dict(weight_bytes: bytes) -> dict:
    buf = io.BytesIO(weight_bytes)
    return torch.load(buf, map_location=device, weights_only=True)


# ── Node communication ────────────────────────────────────────────────────────

def _probe_nodes() -> list[str]:
    """
    Ping all registered client nodes.
    Returns the list of URLs that responded successfully.
    Nodes that are unreachable are excluded from the round.
    """
    alive = []
    for url in CLIENT_NODES:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                info = resp.json()
                print(f"  ✅ {url}  —  images={info.get('num_images', '?')}, "
                      f"device={info.get('device', '?')}")
                alive.append(url)
            else:
                print(f"  ⚠️  {url}  —  HTTP {resp.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"  ❌ {url}  —  unreachable (skipping)")
        except Exception as exc:
            print(f"  ❌ {url}  —  error: {exc}")
    return alive


def _dispatch_train_round(
    node_url: str,
    global_weight_bytes: bytes,
) -> dict | None:
    """
    Send global weights to a client node and collect the updated state_dict.
    Returns a deserialized state_dict dict, or None on failure.
    """
    try:
        resp = requests.post(
            f"{node_url}/train_round",
            files={"weights": ("global_weights.pth", global_weight_bytes,
                               "application/octet-stream")},
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            return _deserialize_state_dict(resp.content)
        else:
            print(f"    ⚠️  {node_url} returned HTTP {resp.status_code}: {resp.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        print(f"    ❌ {node_url} timed out after {TIMEOUT}s.")
        return None
    except Exception as exc:
        print(f"    ❌ {node_url} error: {exc}")
        return None


# ── FedAvg aggregation ────────────────────────────────────────────────────────

def federated_average(client_state_dicts: list[dict]) -> dict:
    """
    Compute the parameter-wise arithmetic mean across all client state_dicts.

        θ_global ← (1/N) * Σ θ_client_i

    All client models must share identical architecture (same key set).
    """
    if not client_state_dicts:
        raise ValueError("Cannot average an empty list of state_dicts.")

    n       = len(client_state_dicts)
    avg_sd  = {}

    # Use the first dict's keys as the reference
    for key in client_state_dicts[0]:
        # Stack all client tensors for this layer and average
        stacked = torch.stack(
            [sd[key].float() for sd in client_state_dicts],
            dim=0,
        )
        avg_sd[key] = stacked.mean(dim=0)

    return avg_sd


# ── Main orchestration loop ───────────────────────────────────────────────────

def run_federated_training() -> None:
    print(f"\n{'='*65}")
    print("FEDERATED LEARNING ORCHESTRATOR")
    print(f"Rounds    : {NUM_ROUNDS}")
    print(f"Clients   : {CLIENT_NODES}")
    print(f"Output    : {OUTPUT_PATH}")
    print(f"Device    : {device}")
    print("=" * 65)

    # Verify node availability before starting
    print("\nProbing client nodes...")
    alive_nodes = _probe_nodes()
    if not alive_nodes:
        raise RuntimeError(
            "No client nodes are reachable. "
            "Start the hospital nodes before running the orchestrator."
        )
    print(f"\n{len(alive_nodes)}/{len(CLIENT_NODES)} nodes available.")

    # Initialise the global model
    global_model = _build_global_model()
    print("\nGlobal model initialised (EfficientNet-B0, IMAGENET1K_V1 weights).")

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n{'─'*65}")
        print(f"ROUND {round_num}/{NUM_ROUNDS}")
        print("─" * 65)

        # 1. Serialise the current global weights
        global_weight_bytes = _serialize_state_dict(global_model.state_dict())
        print(f"  Global weights serialised "
              f"({len(global_weight_bytes)/1024:.1f} KB)")

        # 2. Dispatch to all alive nodes and collect responses
        client_state_dicts = []
        for node_url in alive_nodes:
            print(f"  → Dispatching to {node_url} ...")
            result = _dispatch_train_round(node_url, global_weight_bytes)
            if result is not None:
                client_state_dicts.append(result)
                print(f"     ✅ Received updated weights from {node_url}")
            else:
                print(f"     ⚠️  No usable response from {node_url} — excluded from avg.")

        if not client_state_dicts:
            print(f"  ❌ Round {round_num} failed: no client responses. Skipping aggregation.")
            continue

        # 3. FedAvg aggregation
        print(f"\n  Aggregating {len(client_state_dicts)} client update(s) via FedAvg...")
        averaged_state_dict = federated_average(client_state_dicts)
        global_model.load_state_dict(averaged_state_dict)
        print("  ✅ Global model updated with averaged weights.")

        # 4. Persist checkpoint after every round
        torch.save(
            {
                "round":            round_num,
                "model_state_dict": global_model.state_dict(),
                "num_clients":      len(client_state_dicts),
                "classes":          ["NORMAL", "PNEUMONIA"],
            },
            str(OUTPUT_PATH),
        )
        print(f"  💾 Checkpoint saved → {OUTPUT_PATH}")

    print(f"\n{'='*65}")
    print(f"✅ Federated training complete.  {NUM_ROUNDS} rounds executed.")
    print(f"   Final model: {OUTPUT_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    run_federated_training()
