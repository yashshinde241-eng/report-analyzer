"""
xray_transforms.py — Shared Augmentation & Preprocessing Pipelines
====================================================================
Single source of truth for all image transforms used across training
(train_model.py) and inference (simple_backend.py).

Technique 1 — CLAHE preprocessing
Technique 2 — Albumentations augmentation pipeline (training)
Technique 3 — Label smoothing / cosine annealing live in train_model.py
Technique 4 — TTA inference pipeline lives here and in simple_backend.py

Design notes
------------
- albumentations operates on numpy uint8 HWC arrays.
- torchvision transforms operate on PIL Images or tensors.
- All pipelines normalise with ImageNet stats (the EfficientNet backbone
  was pre-trained on ImageNet, so these stats remain correct).
- CLAHE is applied as a deterministic preprocessing step BEFORE augmentation
  during training and BEFORE inference at runtime.
"""

import io

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from albumentations.pytorch import ToTensorV2
from PIL import Image

# ── ImageNet normalisation constants ─────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


# =============================================================================
# TECHNIQUE 1 — CLAHE Preprocessing
# =============================================================================
# Contrast Limited Adaptive Histogram Equalisation normalises local contrast
# across the image, compensating for differences in X-ray exposure and scanner
# calibration between hospitals and equipment manufacturers.
#
# Why adaptive (CLAHE) instead of global HE?
#   Global histogram equalisation overamplifies noise in already-bright regions.
#   CLAHE divides the image into tiles (tileGridSize) and equalises each tile
#   independently, then clips contrast gain at clipLimit to suppress noise.
#
# clipLimit=2.0 : moderate contrast boost; higher values = more aggressive
# tileGridSize=(8,8) : 8×8 tiles is standard for chest X-ray literature

def apply_clahe(image_bytes: bytes) -> np.ndarray:
    """
    Apply CLAHE to raw image bytes and return an RGB numpy array (uint8, HWC).

    Steps
    -----
    1. Decode bytes → grayscale (X-rays are inherently single-channel).
    2. Apply CLAHE to the single luminance channel.
    3. Convert back to 3-channel RGB so the EfficientNet backbone (which
       expects 3 channels) can consume it unchanged.

    Args:
        image_bytes: Raw bytes of a PNG or JPEG chest X-ray.

    Returns:
        np.ndarray of shape (H, W, 3), dtype uint8, RGB channel order.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    gray  = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        raise ValueError("cv2 could not decode the image bytes.")

    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # Stack to 3-channel RGB
    rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    return rgb  # shape: (H, W, 3), uint8


# =============================================================================
# TECHNIQUE 2 — Albumentations Training Augmentation Pipeline
# =============================================================================
# This pipeline is deliberately aggressive to force the model to learn
# anatomy-based features rather than scanner/positioning artefacts.
#
# Transform rationale
# -------------------
# ElasticTransform  — simulates chest wall deformation due to patient breathing
#                     during image acquisition; alpha controls displacement
#                     magnitude, sigma controls smoothness of the field.
# GridDistortion    — simulates lens/scanner geometric distortion.
# RandomBrightnessContrast — different scanner calibrations produce different
#                     exposure levels; ±30% covers most real-world variance.
# GaussNoise        — models electronic noise in low-dose acquisition modes.
# CLAHE (p=0.4)     — randomly apply during training so the model learns to
#                     handle both raw and CLAHE-processed inputs at test time.
# CoarseDropout     — occludes small rectangular patches, simulating foreign
#                     objects (cables, tubes), body parts obscuring the field,
#                     or partially clipped images.
# RandomRotate90    — occasional 90° rotation handles non-standard patient
#                     positioning and scanner orientation variants.
# HorizontalFlip    — standard left/right symmetry augmentation.

TRAIN_TRANSFORM = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.2),

    # Geometric distortions — simulate scanner/positioning artefacts
    A.ElasticTransform(
        alpha=120,   # displacement field magnitude
        sigma=6,     # smoothness of the displacement field
        p=0.3,
    ),
    A.GridDistortion(
        num_steps=5,
        distort_limit=0.3,
        p=0.3,
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.5,
    ),

    # Noise — models electronic noise at low acquisition doses
    A.GaussNoise(
        std_range=(0.02, 0.12),
        p=0.4,
    ),

    # CLAHE — randomly applied so model handles both processed/raw images
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),

    # Occlusion — foreign objects, partial clips, equipment artefacts
    A.CoarseDropout(
        num_holes_range=(1, 6),
        hole_height_range=(8, 24),
        hole_width_range=(8, 24),
        fill=0,
        p=0.3,
    ),

    A.Normalize(mean=_MEAN, std=_STD),
    ToTensorV2(),
])


# Validation / test-set transform — deterministic, no augmentation.
# CLAHE IS applied here because we always preprocess real-world images with it.
VAL_TRANSFORM = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.Normalize(mean=_MEAN, std=_STD),
    ToTensorV2(),
])


# =============================================================================
# TECHNIQUE 4 — Test-Time Augmentation (TTA) Inference Pipeline
# =============================================================================
# At inference, instead of a single forward pass, we run several deterministic
# augmented views of the same image and average the softmax outputs.
#
# Why this helps
# --------------
# A single prediction is sensitive to exact pixel positioning and brightness.
# Averaging over multiple plausible views reduces this variance and has been
# shown empirically to improve top-1 accuracy by 2–4% on medical imaging
# benchmarks without any retraining.
#
# View set (4 passes by default)
# --------------------------------
# 1. Baseline (CLAHE + resize + normalise)
# 2. Horizontal flip   — left/right positional variance
# 3. Rotate +10°       — slight angulation during scan
# 4. Rotate −10°       — slight angulation during scan
#
# Additional views (if tta_level='full', 8 passes)
# ------------------------------------------------
# 5. Brightness +15%   — over-exposed scanner
# 6. Brightness −15%   — under-exposed scanner
# 7. Contrast +20%
# 8. Contrast −20%

def _tta_transforms_fast() -> list:
    """4-pass TTA — fast, good for interactive / API use."""
    base = [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ]

    flip_base = [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ]

    rot_pos = [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Rotate(limit=(10, 10), p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ]

    rot_neg = [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Rotate(limit=(-10, -10), p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ]

    return [
        A.Compose(base),
        A.Compose(flip_base),
        A.Compose(rot_pos),
        A.Compose(rot_neg),
    ]


def _tta_transforms_full() -> list:
    """8-pass TTA — thorough, for batch/offline processing."""
    fast = _tta_transforms_fast()

    bright_up = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.RandomBrightnessContrast(
            brightness_limit=(0.15, 0.15), contrast_limit=0, p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])
    bright_dn = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.15, -0.15), contrast_limit=0, p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])
    contrast_up = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.RandomBrightnessContrast(
            brightness_limit=0, contrast_limit=(0.2, 0.2), p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])
    contrast_dn = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.RandomBrightnessContrast(
            brightness_limit=0, contrast_limit=(-0.2, -0.2), p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])

    return fast + [bright_up, bright_dn, contrast_up, contrast_dn]


# Public accessor used by simple_backend.py and anywhere else needing TTA.
def get_tta_transforms(level: str = "fast") -> list:
    """
    Return the list of albumentations Compose pipelines for TTA.

    Args:
        level: "fast" (4 passes, ~40 ms/image on CPU) or
               "full" (8 passes, ~80 ms/image on CPU).

    Returns:
        List of A.Compose objects, one per augmentation view.
    """
    if level == "full":
        return _tta_transforms_full()
    return _tta_transforms_fast()


def run_tta_inference(
    model: torch.nn.Module,
    image_bytes: bytes,
    device: torch.device,
    tta_level: str = "fast",
) -> dict:
    """
    Full TTA inference pipeline.

    1. Apply CLAHE to raw bytes → numpy RGB array.
    2. Run each TTA transform over the array.
    3. Forward-pass each view through the model.
    4. Average softmax probabilities across all views.
    5. Return structured prediction dict (all metrics on 0–100 scale).

    Args:
        model      : Loaded EfficientNet-B0 in eval mode.
        image_bytes: Raw PNG/JPEG bytes from an upload.
        device     : torch.device to run inference on.
        tta_level  : "fast" (4 passes) or "full" (8 passes).

    Returns:
        dict with keys:
            prediction    : "NORMAL" | "PNEUMONIA"
            confidence    : float 0–100
            detected      : bool
            severity_score: float 0–100
            tta_passes    : int — number of augmented views used
            normal_prob   : float 0–100 — averaged P(NORMAL)
            pneumonia_prob: float 0–100 — averaged P(PNEUMONIA)
    """
    # Step 1: CLAHE preprocessing → numpy RGB (H,W,3) uint8
    rgb_array = apply_clahe(image_bytes)

    # Step 2: Collect all augmented views as tensors
    transforms_list = get_tta_transforms(tta_level)
    all_probs: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for tfm in transforms_list:
            result = tfm(image=rgb_array)
            tensor = result["image"].unsqueeze(0).to(device)  # (1,3,H,W)
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)             # (1,2)
            all_probs.append(probs)

    # Step 3: Average across all TTA views
    # Shape: (num_passes, 1, 2) → mean over dim=0 → (1, 2)
    stacked   = torch.stack(all_probs, dim=0)   # (N, 1, 2)
    avg_probs = stacked.mean(dim=0)              # (1, 2)

    normal_p    = avg_probs[0, 0].item()   # P(NORMAL)
    pneumonia_p = avg_probs[0, 1].item()   # P(PNEUMONIA)

    detected   = pneumonia_p >= normal_p
    confidence = pneumonia_p if detected else normal_p

    # Severity: high = urgent review needed
    # Pneumonia detected + high confidence → high severity
    # Normal detected + low confidence → also warrants review
    raw_severity   = pneumonia_p if detected else (1.0 - normal_p)
    severity_score = round(raw_severity * 100, 2)
    confidence_pct = round(confidence * 100, 2)

    return {
        "prediction":     "PNEUMONIA" if detected else "NORMAL",
        "confidence":     confidence_pct,       # 0–100
        "detected":       detected,
        "severity_score": severity_score,        # 0–100
        "tta_passes":     len(transforms_list),
        "normal_prob":    round(normal_p * 100, 2),
        "pneumonia_prob": round(pneumonia_p * 100, 2),
    }
