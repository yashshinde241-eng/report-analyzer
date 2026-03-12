import json
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================
# CONFIGURATION
# ============================================
DATA_PATH = r"C:\Users\LOQ\OneDrive\Desktop\report-analyzer\data\text_reports.json"
MODEL_SAVE_PATH = r"C:\Users\LOQ\OneDrive\Desktop\report-analyzer\models\text_model.pkl"
VECTORIZER_SAVE_PATH = r"C:\Users\LOQ\OneDrive\Desktop\report-analyzer\models\vectorizer.pkl"

# ============================================
# LOAD DATA
# ============================================
print("="*50)
print("DIABETES TEXT CLASSIFIER TRAINING")
print("="*50)

print("\nLoading dataset...")
with open(DATA_PATH, 'r') as f:
    dataset = json.load(f)

texts = [d['text'] for d in dataset]
labels = [d['label'] for d in dataset]

print(f"✅ Loaded {len(texts)} reports")
print(f"   Diabetic : {sum(labels)}")
print(f"   Normal   : {len(labels) - sum(labels)}")

# ============================================
# SPLIT DATA
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.2,       # 80% train, 20% test
    random_state=42,
    stratify=labels      # Keep class balance in both splits
)

print(f"\nData split:")
print(f"   Train : {len(X_train)} reports")
print(f"   Test  : {len(X_test)} reports")

# ============================================
# CONVERT TEXT TO NUMBERS (TF-IDF)
# ============================================
print("\nConverting text to TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=5000,      # Use top 5000 words
    ngram_range=(1, 2),     # Use single words and pairs
    stop_words='english',   # Remove common words like 'the', 'is'
    lowercase=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"✅ Feature matrix shape: {X_train_tfidf.shape}")

# ============================================
# TRAIN MODEL
# ============================================
print("\nTraining Logistic Regression classifier...")
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    C=1.0
)
model.fit(X_train_tfidf, y_train)
print("✅ Training complete!")

# ============================================
# EVALUATE MODEL
# ============================================
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# Predictions
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Normal', 'Diabetic']))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"              Predicted")
print(f"              Normal  Diabetic")
print(f"Actual Normal  {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"Actual Diabetic{cm[1][0]:4d}    {cm[1][1]:4d}")

# ============================================
# TEST WITH SAMPLE REPORTS
# ============================================
print("\n" + "="*50)
print("SAMPLE PREDICTIONS")
print("="*50)

sample_diabetic = """
MEDICAL REPORT
Patient: John Doe, Age: 52
FINDINGS:
- fasting blood glucose level is 245 mg/dL
- HbA1c level measured at 9.2%
- patient shows signs of hyperglycemia
- insulin resistance detected
Diagnosis: Type 2 diabetes mellitus detected.
"""

sample_normal = """
MEDICAL REPORT
Patient: Jane Smith, Age: 35
FINDINGS:
- blood pressure within normal limits
- blood sugar levels within normal range
- no evidence of diabetes
- all results within normal limits
Impression: Patient in good health.
"""

for label, sample in [("DIABETIC", sample_diabetic), ("NORMAL", sample_normal)]:
    sample_tfidf = vectorizer.transform([sample])
    prediction = model.predict(sample_tfidf)[0]
    probability = model.predict_proba(sample_tfidf)[0]
    confidence = max(probability) * 100
    predicted_label = "Diabetic" if prediction == 1 else "Normal"
    print(f"\nSample ({label}):")
    print(f"   Predicted  : {predicted_label}")
    print(f"   Confidence : {confidence:.2f}%")
    print(f"   Correct    : {'✅' if (label == 'DIABETIC') == (prediction == 1) else '❌'}")

# ============================================
# SAVE MODEL
# ============================================
print("\n" + "="*50)
print("SAVING MODEL")
print("="*50)

joblib.dump(model, MODEL_SAVE_PATH)
joblib.dump(vectorizer, VECTORIZER_SAVE_PATH)

print(f"✅ Model saved     : {MODEL_SAVE_PATH}")
print(f"✅ Vectorizer saved: {VECTORIZER_SAVE_PATH}")
print(f"\n🎉 Text classifier ready to use!")
print("="*50)