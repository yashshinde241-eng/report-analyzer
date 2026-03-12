import random
import json
import os

# ============================================
# CONFIGURATION
# ============================================
OUTPUT_PATH = r"C:\Users\LOQ\OneDrive\Desktop\report-analyzer\data\text_reports.json"
NUM_SAMPLES = 1000  # 500 diabetic + 500 normal

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ============================================
# REPORT TEMPLATES
# ============================================

# Diabetic report components
diabetic_findings = [
    "fasting blood glucose level is {glucose} mg/dL",
    "HbA1c level measured at {hba1c}%",
    "patient shows signs of hyperglycemia",
    "blood sugar levels consistently elevated",
    "insulin resistance detected",
    "glucose tolerance test abnormal",
    "random blood glucose {glucose} mg/dL",
    "patient diagnosed with type 2 diabetes",
    "diabetic retinopathy observed",
    "elevated fasting glucose levels noted",
    "hemoglobin A1c above normal range",
    "patient on insulin therapy",
    "metformin prescribed for blood sugar control",
    "postprandial glucose elevated at {glucose} mg/dL",
    "diabetic neuropathy symptoms present",
]

normal_findings = [
    "fasting blood glucose level is {glucose} mg/dL",
    "HbA1c level measured at {hba1c}%",
    "blood sugar levels within normal range",
    "glucose tolerance test normal",
    "no signs of insulin resistance",
    "random blood glucose {glucose} mg/dL",
    "no evidence of diabetes",
    "cholesterol levels normal",
    "blood pressure within normal limits",
    "thyroid function normal",
    "complete blood count normal",
    "liver function tests normal",
    "kidney function normal",
    "no abnormalities detected",
    "patient in good health",
]

patient_info = [
    "Patient: {name}, Age: {age}",
    "Name: {name}, DOB: {dob}",
    "Patient ID: {pid}, Age: {age}",
]

names = ["John Smith", "Mary Johnson", "Robert Davis", "Lisa Wilson",
         "James Brown", "Sarah Miller", "Michael Taylor", "Emily Anderson",
         "David Martinez", "Jennifer Thomas", "William Jackson", "Jessica White"]

conclusions_diabetic = [
    "Impression: Patient shows clear indicators of diabetes mellitus.",
    "Conclusion: Diabetes confirmed. Immediate treatment recommended.",
    "Assessment: Blood glucose levels indicate uncontrolled diabetes.",
    "Diagnosis: Type 2 diabetes mellitus detected.",
    "Summary: Patient requires diabetes management plan.",
]

conclusions_normal = [
    "Impression: All results within normal limits.",
    "Conclusion: No evidence of metabolic disorders.",
    "Assessment: Patient appears healthy. Routine follow-up recommended.",
    "Diagnosis: No abnormalities detected.",
    "Summary: Normal report. Continue regular checkups.",
]


# ============================================
# REPORT GENERATOR
# ============================================

def generate_diabetic_report():
    """Generate a fake diabetic patient report"""
    name = random.choice(names)
    age = random.randint(35, 75)
    glucose = random.randint(180, 400)  # High glucose
    hba1c = round(random.uniform(7.0, 12.0), 1)  # High HbA1c
    pid = random.randint(10000, 99999)
    dob = f"{random.randint(1, 28)}/{random.randint(1, 12)}/{random.randint(1950, 1990)}"

    # Pick random findings
    num_findings = random.randint(3, 6)
    findings = random.sample(diabetic_findings, num_findings)
    findings = [f.format(glucose=glucose, hba1c=hba1c) for f in findings]

    # Build report
    header = random.choice(patient_info).format(
        name=name, age=age, pid=pid, dob=dob)
    conclusion = random.choice(conclusions_diabetic)

    report = f"""MEDICAL REPORT
==============
{header}
Date: {random.randint(1, 28)}/{random.randint(1, 12)}/2024

FINDINGS:
{chr(10).join(f'- {f}' for f in findings)}

{conclusion}
"""
    return report.strip()


def generate_normal_report():
    """Generate a fake normal patient report"""
    name = random.choice(names)
    age = random.randint(18, 70)
    glucose = random.randint(70, 99)   # Normal glucose
    hba1c = round(random.uniform(4.0, 5.6), 1)  # Normal HbA1c
    pid = random.randint(10000, 99999)
    dob = f"{random.randint(1, 28)}/{random.randint(1, 12)}/{random.randint(1950, 2000)}"

    # Pick random findings
    num_findings = random.randint(3, 6)
    findings = random.sample(normal_findings, num_findings)
    findings = [f.format(glucose=glucose, hba1c=hba1c) for f in findings]

    # Build report
    header = random.choice(patient_info).format(
        name=name, age=age, pid=pid, dob=dob)
    conclusion = random.choice(conclusions_normal)

    report = f"""MEDICAL REPORT
==============
{header}
Date: {random.randint(1, 28)}/{random.randint(1, 12)}/2024

FINDINGS:
{chr(10).join(f'- {f}' for f in findings)}

{conclusion}
"""
    return report.strip()


# ============================================
# GENERATE DATASET
# ============================================
print("="*50)
print("GENERATING SYNTHETIC MEDICAL TEXT DATA")
print("="*50)

dataset = []

# Generate diabetic reports
print(f"\nGenerating {NUM_SAMPLES//2} diabetic reports...")
for _ in range(NUM_SAMPLES // 2):
    report = generate_diabetic_report()
    dataset.append({'text': report, 'label': 1, 'disease': 'diabetic'})

# Generate normal reports
print(f"Generating {NUM_SAMPLES//2} normal reports...")
for _ in range(NUM_SAMPLES // 2):
    report = generate_normal_report()
    dataset.append({'text': report, 'label': 0, 'disease': 'normal'})

# Shuffle dataset
random.shuffle(dataset)

# Save to file
with open(OUTPUT_PATH, 'w') as f:
    json.dump(dataset, f, indent=2)

print(f"\n✅ Dataset saved to: {OUTPUT_PATH}")
print(f"   Total samples : {len(dataset)}")
print(f"   Diabetic      : {sum(1 for d in dataset if d['label'] == 1)}")
print(f"   Normal        : {sum(1 for d in dataset if d['label'] == 0)}")

# Show sample report
print("\n" + "="*50)
print("SAMPLE DIABETIC REPORT:")
print("="*50)
print(dataset[0]['text'] if dataset[0]['label'] == 1 
      else next(d['text'] for d in dataset if d['label'] == 1))

print("\n" + "="*50)
print("SAMPLE NORMAL REPORT:")
print("="*50)
print(next(d['text'] for d in dataset if d['label'] == 0))
print("="*50)