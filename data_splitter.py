"""
data_splitter.py — Hospital Data Isolation Layer
=================================================
Distributes the Kaggle chest X-ray training images across three simulated
hospital partitions while preserving class stratification.

Output directories (created automatically):
    data/hospital_A/NORMAL/       data/hospital_A/PNEUMONIA/
    data/hospital_B/NORMAL/       data/hospital_B/PNEUMONIA/
    data/hospital_C/NORMAL/       data/hospital_C/PNEUMONIA/

Files are *copied* (not moved) so the original dataset remains intact.

Usage:
    python data_splitter.py [--dataset /path/to/chest_xray/train]
    python data_splitter.py  # auto-detects ./data set/chest_xray/train
"""

import argparse
import random
import shutil
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN = ROOT_DIR / "data set" / "chest_xray" / "train"

parser = argparse.ArgumentParser(description="Split chest X-ray data for federated learning")
parser.add_argument(
    "--dataset",
    default=str(DEFAULT_TRAIN),
    help="Path to the chest_xray/train directory (must contain NORMAL/ and PNEUMONIA/).",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducible splits (default: 42).",
)
args = parser.parse_args()

TRAIN_DIR  = Path(args.dataset)
OUTPUT_DIR = ROOT_DIR / "data"
HOSPITALS  = ["hospital_A", "hospital_B", "hospital_C"]
CLASSES    = ["NORMAL", "PNEUMONIA"]
SEED       = args.seed

random.seed(SEED)


def split_evenly(items: list, n: int) -> list[list]:
    """
    Divide *items* into *n* partitions as evenly as possible.
    The list is shuffled in-place before splitting so the distribution
    is random rather than file-system-order dependent.
    """
    random.shuffle(items)
    partitions = [[] for _ in range(n)]
    for idx, item in enumerate(items):
        partitions[idx % n].append(item)
    return partitions


def main() -> None:
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_DIR}\n"
            f"Pass the correct path with --dataset /path/to/chest_xray/train"
        )

    print(f"\n{'='*60}")
    print("FEDERATED DATA SPLITTER")
    print(f"Source  : {TRAIN_DIR}")
    print(f"Output  : {OUTPUT_DIR}")
    print(f"Seed    : {SEED}")
    print("=" * 60)

    grand_total = 0

    for class_name in CLASSES:
        class_dir = TRAIN_DIR / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Expected class folder missing: {class_dir}"
            )

        # Gather all image files for this class
        image_files = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not image_files:
            print(f"  ⚠️  No images found in {class_dir} — skipping.")
            continue

        partitions = split_evenly(image_files, len(HOSPITALS))

        print(f"\nClass: {class_name}  ({len(image_files)} images)")
        for hospital, partition in zip(HOSPITALS, partitions):
            dest_dir = OUTPUT_DIR / hospital / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for src_path in partition:
                shutil.copy2(src_path, dest_dir / src_path.name)

            grand_total += len(partition)
            print(f"  {hospital:12s} → {len(partition):4d} images  →  {dest_dir}")

    print(f"\n{'='*60}")
    print(f"✅ Split complete.  {grand_total} files distributed across {len(HOSPITALS)} nodes.")
    print("=" * 60)

    # Print summary counts for verification
    print("\nVerification counts:")
    for hospital in HOSPITALS:
        counts = {}
        for class_name in CLASSES:
            d = OUTPUT_DIR / hospital / class_name
            counts[class_name] = len(list(d.glob("*"))) if d.exists() else 0
        total = sum(counts.values())
        print(f"  {hospital}: " + "  ".join(f"{k}={v}" for k, v in counts.items())
              + f"  (total={total})")


if __name__ == "__main__":
    main()
