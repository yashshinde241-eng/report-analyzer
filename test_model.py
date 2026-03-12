import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = r"C:\Users\LOQ\OneDrive\Desktop\report-analyzer\models\pneumonia_model.pth"
TEST_IMAGE_PATH = r"C:\Users\LOQ\OneDrive\Desktop\report-analyzer\data set\chest_xray\test\PNEUMONIA\person1_virus_6.jpeg"

# ============================================
# LOAD MODEL
# ============================================
print("\n" + "="*50)
print("Loading model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Rebuild model architecture
model = models.efficientnet_b0(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)

# Load saved weights
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

classes = checkpoint['classes']
saved_accuracy = checkpoint['accuracy']

print(f"Model loaded successfully!")
print(f"Classes: {classes}")
print(f"Saved accuracy: {saved_accuracy:.2f}%")

# ============================================
# PREPARE IMAGE
# ============================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    """Run prediction on a single image"""
    
    print(f"\n{'='*50}")
    print(f"Analyzing image:")
    print(f"{image_path}")
    print("="*50)

    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Run prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    # Get results
    class_name = classes[predicted_class.item()]
    confidence_score = confidence.item() * 100
    detected = class_name == 'PNEUMONIA'

    print(f"\n📊 RESULTS:")
    print(f"   Predicted Class : {class_name}")
    print(f"   Detected        : {detected}")
    print(f"   Confidence      : {confidence_score:.2f}%")
    print(f"\n   NORMAL     probability: {probabilities[0][0].item()*100:.2f}%")
    print(f"   PNEUMONIA  probability: {probabilities[0][1].item()*100:.2f}%")

    return {
        'detected': detected,
        'confidence': round(confidence_score, 2),
        'diseaseName': 'Pneumonia',
        'predictedClass': class_name
    }

# ============================================
# TEST WITH ONE IMAGE
# ============================================
result = predict_image(TEST_IMAGE_PATH)

print(f"\n{'='*50}")
print(f"✅ Final output (this is what Flask will return):")
print(f"   {result}")
print("="*50)