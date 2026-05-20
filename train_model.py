"""
train_model.py — EfficientNet-B0 Pneumonia Classifier Training
================================================================
Techniques integrated
---------------------
  1. CLAHE preprocessing     — via xray_transforms.apply_clahe (in dataset)
  2. Albumentations pipeline — via xray_transforms.TRAIN_TRANSFORM / VAL_TRANSFORM
  3. Label smoothing         — nn.CrossEntropyLoss(label_smoothing=0.1)
     Cosine annealing WR     — CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
  4. TTA at inference        — lives in simple_backend.py / xray_transforms.py

Dataset layout expected (Kaggle chest X-ray):
    <DATASET_PATH>/
        train/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/

Usage:
    python train_model.py [--dataset /path/to/chest_xray] [--epochs 15]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models

from xray_transforms import TRAIN_TRANSFORM, VAL_TRANSFORM, apply_clahe

# ── Argument parsing ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT_DIR / "data set" / "chest_xray"

parser = argparse.ArgumentParser(description="Train pneumonia EfficientNet-B0")
parser.add_argument(
    "--dataset",
    default=str(DEFAULT_DATASET),
    help="Root of the chest_xray directory (contains train/ and test/).",
)
parser.add_argument("--epochs",     type=int,   default=15)
parser.add_argument("--batch-size", type=int,   default=32)
parser.add_argument("--lr",         type=float, default=0.001)
args = parser.parse_args()

DATASET_PATH    = Path(args.dataset)
MODEL_SAVE_PATH = ROOT_DIR / "models" / "pneumonia_model.pth"
BATCH_SIZE      = args.batch_size
EPOCHS          = args.epochs
LEARNING_RATE   = args.lr

MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*55}\nUsing device: {device}\n{'='*55}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM DATASET — applies CLAHE first, then albumentations transform
# ═══════════════════════════════════════════════════════════════════════════════

class XRayDataset(Dataset):
    """
    PyTorch Dataset for chest X-ray images.

    Pipeline per sample
    -------------------
    1. Read raw bytes from disk.
    2. Apply CLAHE (Technique 1) — normalises local contrast per image,
       compensating for scanner exposure differences.
    3. Apply the albumentations transform (Technique 2 during training,
       deterministic VAL_TRANSFORM during validation/test).

    Using raw bytes → CLAHE → albumentations (rather than PIL → torchvision)
    keeps the full pipeline in one place and avoids redundant decode steps.
    """

    CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

    def __init__(self, root_dir: Path, transform):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for label_idx, class_name in enumerate(self.CLASS_NAMES):
            class_dir = root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Expected class directory not found: {class_dir}"
                )
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_path, label_idx))

        if not self.samples:
            raise RuntimeError(f"No images found under {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # Step 1: read raw bytes
        raw_bytes = img_path.read_bytes()

        # Step 2: CLAHE — produces uint8 numpy RGB array (H, W, 3)
        rgb_array = apply_clahe(raw_bytes)

        # Step 3: Albumentations transform → float32 tensor (3, H, W)
        augmented = self.transform(image=rgb_array)
        tensor    = augmented["image"]

        return tensor, label

    @property
    def class_counts(self) -> list[int]:
        """Count samples per class for imbalance handling."""
        counts = [0, 0]
        for _, label in self.samples:
            counts[label] += 1
        return counts


# ═══════════════════════════════════════════════════════════════════════════════
# MIXUP — interpolates pairs of training samples and their labels
# ═══════════════════════════════════════════════════════════════════════════════
# Mixup creates virtual training examples by linearly interpolating between
# two real samples:
#
#     x̃ = λ·x_i + (1−λ)·x_j
#     ỹ = λ·y_i + (1−λ)·y_j
#
# where λ ~ Beta(α, α).  This regularises the model, discourages overconfidence
# on individual training examples, and improves generalisation to domain-shifted
# real-world data.  α=0.4 is the standard value for image classification.

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply Mixup to a batch.

    Args:
        x    : Input batch tensor (B, C, H, W).
        y    : Label tensor (B,).
        alpha: Beta distribution concentration parameter.

    Returns:
        mixed_x : Interpolated inputs.
        y_a     : Original labels.
        y_b     : Shuffled labels.
        lam     : Interpolation coefficient λ.
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a = y
    y_b = y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute the Mixup loss: λ·L(pred,y_a) + (1−λ)·L(pred,y_b)."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASETS & LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

print("Loading datasets with CLAHE + Albumentations pipeline...")
train_dataset = XRayDataset(DATASET_PATH / "train", transform=TRAIN_TRANSFORM)
test_dataset  = XRayDataset(DATASET_PATH / "test",  transform=VAL_TRANSFORM)

print(f"Train: {len(train_dataset)} images")
print(f"Test : {len(test_dataset)} images")

# Weighted sampler to handle NORMAL / PNEUMONIA class imbalance
counts  = train_dataset.class_counts
print(f"\nClass counts — NORMAL: {counts[0]}, PNEUMONIA: {counts[1]}")

sample_weights = [1.0 / counts[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

print("\nLoading pretrained EfficientNet-B0...")
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# Replace the classifier head with a regularised binary head
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(p=0.2),
    nn.Linear(512, 2),
)
model = model.to(device)
print("Model ready.")


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 3a — LABEL SMOOTHING LOSS
# ═══════════════════════════════════════════════════════════════════════════════
# Instead of training against hard targets [0,1] / [1,0], label smoothing
# trains against softened targets [ε/K, 1−ε(K−1)/K] = [0.05, 0.95].
#
# This prevents the model from becoming overconfident on training labels
# (many of which may be noisy in a real radiology dataset) and empirically
# improves calibration — confidence scores more closely track true accuracy.
#
# ε = 0.1 is the standard value from the original paper (Szegedy et al. 2016).

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 3b — COSINE ANNEALING WITH WARM RESTARTS
# ═══════════════════════════════════════════════════════════════════════════════
# The learning rate follows a cosine curve from η_max down to η_min, then
# resets.  Restarts help the optimiser escape local minima that arise from
# dataset-specific artefacts (e.g., scanner brand patterns).
#
# T_0=10  : first restart period is 10 epochs
# T_mult=2: each subsequent period doubles (10 → 20 → 40 …)
# eta_min : floor LR to avoid complete learning stoppage

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2,
    eta_min=1e-6,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nStarting training — {EPOCHS} epochs, label smoothing=0.1, "
      f"cosine warm restarts (T_0=10)\n{'='*55}")

best_accuracy = 0.0
USE_MIXUP     = True   # set False to disable Mixup for ablation studies

for epoch in range(EPOCHS):
    epoch_start = time.time()

    # ── Training phase ────────────────────────────────────────────────────────
    model.train()
    train_loss    = 0.0
    train_correct = 0
    train_total   = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        if USE_MIXUP:
            # Apply Mixup augmentation to this batch
            mixed_images, y_a, y_b, lam = mixup_data(images, labels, alpha=0.4)
            optimizer.zero_grad()
            outputs = model(mixed_images)
            loss    = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss  += loss.item()
        _, predicted = outputs.max(1)
        train_total  += labels.size(0)
        # For Mixup accuracy, compare against original (non-mixed) labels
        train_correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 50 == 0:
            running_loss = train_loss / (batch_idx + 1)
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} "
                  f"| Loss: {running_loss:.4f} "
                  f"| LR: {scheduler.get_last_lr()[0]:.6f}")

    # ── Cosine scheduler step (per epoch) ────────────────────────────────────
    scheduler.step(epoch)

    train_acc = 100.0 * train_correct / train_total

    # ── Evaluation phase ──────────────────────────────────────────────────────
    model.eval()
    test_correct = 0
    test_total   = 0
    tp = fp = tn = fn = 0   # for sensitivity / specificity

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs   = model(images)
            _, predicted = outputs.max(1)
            test_total   += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            # Per-class counts for clinical metrics
            for pred, lbl in zip(predicted, labels):
                p, l = pred.item(), lbl.item()
                if l == 1 and p == 1: tp += 1   # pneumonia correctly detected
                if l == 0 and p == 1: fp += 1   # false alarm
                if l == 0 and p == 0: tn += 1   # normal correctly cleared
                if l == 1 and p == 0: fn += 1   # missed pneumonia

    test_acc    = 100.0 * test_correct / test_total
    sensitivity = 100.0 * tp / max(tp + fn, 1)   # recall for PNEUMONIA
    specificity = 100.0 * tn / max(tn + fp, 1)   # recall for NORMAL
    epoch_time  = time.time() - epoch_start

    print(f"\nEpoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s) "
          f"| Train: {train_acc:.2f}%  Test: {test_acc:.2f}%"
          f"\n  Sensitivity (pneumonia recall): {sensitivity:.2f}%"
          f"  Specificity (normal recall): {specificity:.2f}%")

    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(
            {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy":             best_accuracy,
                "sensitivity":          sensitivity,
                "specificity":          specificity,
                "classes":              ["NORMAL", "PNEUMONIA"],
                "techniques":           [
                    "CLAHE preprocessing",
                    "Albumentations augmentation",
                    "Label smoothing (0.1)",
                    "Cosine annealing warm restarts",
                    "Mixup (alpha=0.4)",
                ],
            },
            str(MODEL_SAVE_PATH),
        )
        print(f"  ✅ New best saved — accuracy: {best_accuracy:.2f}%")
    print("-" * 55)

print(f"\n{'='*55}")
print(f"Training complete.")
print(f"Best test accuracy : {best_accuracy:.2f}%")
print(f"Model saved to     : {MODEL_SAVE_PATH}")
print("=" * 55)
