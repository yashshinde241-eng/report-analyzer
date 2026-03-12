import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import os
import time

# ============================================
# CONFIGURATION
# ============================================
DATASET_PATH = r"C:\Users\LOQ\OneDrive\Desktop\report-analyzer\data set\chest_xray"
MODEL_SAVE_PATH = r"C:\Users\LOQ\OneDrive\Desktop\report-analyzer\models\pneumonia_model.pth"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# ============================================
# SETUP
# ============================================
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*50}")
print(f"Using device: {device}")
print(f"{'='*50}\n")

# ============================================
# DATA TRANSFORMS
# ============================================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================================
# LOAD DATA
# ============================================
print("Loading datasets...")
train_dataset = datasets.ImageFolder(
    os.path.join(DATASET_PATH, 'train'),
    transform=train_transform
)
test_dataset = datasets.ImageFolder(
    os.path.join(DATASET_PATH, 'test'),
    transform=test_transform
)

print(f"Classes: {train_dataset.classes}")
print(f"Train size: {len(train_dataset)}")
print(f"Test size : {len(test_dataset)}")

# ============================================
# HANDLE CLASS IMBALANCE
# ============================================
# Count samples per class
class_counts = [0, 0]
for _, label in train_dataset:
    class_counts[label] += 1

print(f"\nClass counts - NORMAL: {class_counts[0]}, PNEUMONIA: {class_counts[1]}")

# Give higher weight to the minority class (NORMAL)
weights = [1.0 / class_counts[label] for _, label in train_dataset]
sampler = WeightedRandomSampler(weights, len(weights))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ============================================
# BUILD MODEL
# ============================================
print("\nLoading pretrained EfficientNet...")
model = models.efficientnet_b0(weights='IMAGENET1K_V1')

# Replace the final layer for binary classification
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)
model = model.to(device)
print("Model ready!")

# ============================================
# TRAINING SETUP
# ============================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# ============================================
# TRAINING LOOP
# ============================================
print(f"\nStarting training for {EPOCHS} epochs...")
print("="*50)

best_accuracy = 0.0

for epoch in range(EPOCHS):
    start_time = time.time()

    # --- Training phase ---
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} "
                  f"| Loss: {train_loss/(batch_idx+1):.3f}")

    train_accuracy = 100. * train_correct / train_total

    # --- Evaluation phase ---
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_accuracy = 100. * test_correct / test_total
    epoch_time = time.time() - start_time
    scheduler.step()

    print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
    print(f"  Train Accuracy : {train_accuracy:.2f}%")
    print(f"  Test Accuracy  : {test_accuracy:.2f}%")
    print(f"  Time           : {epoch_time:.1f}s")
    print("-"*50)

    # Save best model
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': best_accuracy,
            'classes': train_dataset.classes
        }, MODEL_SAVE_PATH)
        print(f"  ✅ New best model saved! Accuracy: {best_accuracy:.2f}%")

print(f"\n{'='*50}")
print(f"Training Complete!")
print(f"Best Test Accuracy: {best_accuracy:.2f}%")
print(f"Model saved to: {MODEL_SAVE_PATH}")
print(f"{'='*50}")