"""
test_model.py — Standalone Pneumonia Model Inference Test
==========================================================
Loads the trained checkpoint and runs a single-image prediction.

Usage:
    python test_model.py --image /path/to/chest_xray.jpg
    python test_model.py --image /path/to/chest_xray.jpg --model /path/to/other.pth
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ── CLI args ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Test the pneumonia EfficientNet-B0 model")
parser.add_argument(
    "--model",
    default=str(ROOT_DIR / "models" / "pneumonia_model.pth"),
    help="Path to the .pth checkpoint file.",
)
parser.add_argument(
    "--image",
    required=True,
    help="Path to the chest X-ray image to classify.",
)
args = parser.parse_args()

MODEL_PATH = Path(args.model)
IMAGE_PATH = Path(args.image)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*50}\nUsing device: {device}\n{'='*50}")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

classes        = checkpoint["classes"]
saved_accuracy = checkpoint.get("accuracy", "N/A")

print(f"Classes        : {classes}")
print(f"Saved accuracy : {saved_accuracy}")

# ── Inference transform ───────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Predict ───────────────────────────────────────────────────────────────────
print(f"\nAnalyzing: {IMAGE_PATH}")
image  = Image.open(str(IMAGE_PATH)).convert("RGB")
tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs       = model(tensor)
    probabilities = torch.softmax(outputs, dim=1)
    conf, pred_cls = torch.max(probabilities, 1)

class_name  = classes[pred_cls.item()]
confidence  = conf.item() * 100
detected    = class_name == "PNEUMONIA"
severity    = confidence if detected else (100.0 - confidence)

print(f"\n📊 RESULTS:")
print(f"   Predicted Class  : {class_name}")
print(f"   Detected         : {detected}")
print(f"   Confidence       : {confidence:.2f}%")
print(f"   Severity Score   : {severity:.2f}%")
print(f"\n   NORMAL    probability: {probabilities[0][0].item()*100:.2f}%")
print(f"   PNEUMONIA probability: {probabilities[0][1].item()*100:.2f}%")

print(f"\n{'='*50}")
print(f"✅ Final output:")
print({
    "detected":      detected,
    "confidence":    round(confidence, 2),
    "severity_score": round(severity, 2),
    "disease_name":  "Pneumonia",
    "predicted_class": class_name,
})
print("=" * 50)
