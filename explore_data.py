import os
from pathlib import Path

# Dataset path
DATASET_PATH = r"C:\Users\LOQ\OneDrive\Desktop\report-analyzer\data set\chest_xray"

def count_images(folder_path):
    """Count images in a folder"""
    extensions = {'.jpg', '.jpeg', '.png'}
    return sum(1 for f in Path(folder_path).iterdir() 
               if f.suffix.lower() in extensions)

print("=" * 50)
print("CHEST X-RAY DATASET EXPLORATION")
print("=" * 50)

# Check each split
for split in ['train', 'val', 'test']:
    split_path = os.path.join(DATASET_PATH, split)
    
    normal_path = os.path.join(split_path, 'NORMAL')
    pneumonia_path = os.path.join(split_path, 'PNEUMONIA')
    
    normal_count = count_images(normal_path)
    pneumonia_count = count_images(pneumonia_path)
    total = normal_count + pneumonia_count
    
    print(f"\n📁 {split.upper()}")
    print(f"   NORMAL    : {normal_count} images")
    print(f"   PNEUMONIA : {pneumonia_count} images")
    print(f"   TOTAL     : {total} images")

print("\n" + "=" * 50)
print("✅ Dataset check complete!")
print("=" * 50)