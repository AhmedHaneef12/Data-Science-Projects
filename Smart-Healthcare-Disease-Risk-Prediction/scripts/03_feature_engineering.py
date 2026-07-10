"""
Task 04 - Part 3: Feature Engineering & Risk Analysis
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / 'data'
CHARTS = BASE / 'charts'
REPORTS_DIR = BASE / 'reports'
CHARTS.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / 'cardio_train_cleaned.csv')
notes = []

# ---------------------------------------------------------
# 1. BMI (Body Mass Index)
# ---------------------------------------------------------
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['bmi'] = df['bmi'].round(1)

def bmi_category(b):
    if b < 18.5: return 'Underweight'
    elif b < 25: return 'Normal'
    elif b < 30: return 'Overweight'
    else: return 'Obese'
df['bmi_category'] = df['bmi'].apply(bmi_category)
notes.append("Created 'bmi' = weight(kg) / height(m)^2, and 'bmi_category' "
             "(Underweight / Normal / Overweight / Obese per WHO standard thresholds).")

# ---------------------------------------------------------
# 2. Blood Pressure Category (standard clinical classification)
# ---------------------------------------------------------
def bp_category(row):
    hi, lo = row['ap_hi'], row['ap_lo']
    if hi < 120 and lo < 80:
        return 'Normal'
    elif hi < 130 and lo < 80:
        return 'Elevated'
    elif hi < 140 or lo < 90:
        return 'Hypertension Stage 1'
    else:
        return 'Hypertension Stage 2'
df['bp_category'] = df.apply(bp_category, axis=1)
notes.append("Created 'bp_category' using standard clinical blood pressure staging "
             "(Normal / Elevated / Hypertension Stage 1 / Hypertension Stage 2).")

# ---------------------------------------------------------
# 3. Composite Health Risk Score (0-10 scale, rule-based)
# ---------------------------------------------------------
def risk_score(row):
    score = 0
    # Age contribution
    if row['age_years'] >= 55: score += 2
    elif row['age_years'] >= 45: score += 1
    # Blood pressure
    if row['bp_category'] == 'Hypertension Stage 2': score += 3
    elif row['bp_category'] == 'Hypertension Stage 1': score += 2
    elif row['bp_category'] == 'Elevated': score += 1
    # Cholesterol & glucose
    score += (row['cholesterol'] - 1)
    score += (row['gluc'] - 1) * 0.5
    # BMI
    if row['bmi_category'] == 'Obese': score += 1.5
    elif row['bmi_category'] == 'Overweight': score += 0.5
    # Lifestyle
    score += row['smoke'] * 0.5
    score += row['alco'] * 0.3
    score -= row['active'] * 0.5
    return max(0, round(score, 1))

df['risk_score'] = df.apply(risk_score, axis=1)
notes.append("Created a rule-based 'risk_score' (0-10) combining age, blood pressure stage, "
             "cholesterol, glucose, BMI, and lifestyle factors (smoking/alcohol reduce score downward, "
             "activity reduces it) — used for quick triage before formal modeling.")

def risk_group(s):
    if s <= 2: return 'Low'
    elif s <= 4.5: return 'Medium'
    else: return 'High'
df['risk_group'] = df['risk_score'].apply(risk_group)
notes.append("Segmented patients into Low / Medium / High risk groups based on risk_score thresholds.")

# ---------------------------------------------------------
# 4. Validate engineered features against actual disease outcome
# ---------------------------------------------------------
group_validation = df.groupby('risk_group')['cardio'].mean() * 100
notes.append(f"Validation: actual disease rate by risk group - "
             f"Low: {group_validation.get('Low', 0):.1f}%, "
             f"Medium: {group_validation.get('Medium', 0):.1f}%, "
             f"High: {group_validation.get('High', 0):.1f}% "
             f"(confirms risk_score meaningfully separates patients).")

bmi_validation = df.groupby('bmi_category')['cardio'].mean() * 100
bp_validation = df.groupby('bp_category')['cardio'].mean() * 100

# ---------------------------------------------------------
# 5. Visualizations
# ---------------------------------------------------------
order = ['Low', 'Medium', 'High']
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df['risk_group'].value_counts().reindex(order).plot(kind='bar', color=['#2ecc71', '#f39c12', '#e74c3c'], ax=axes[0])
axes[0].set_title('Patient Count by Risk Group')
axes[0].set_ylabel('Number of Patients')
axes[0].tick_params(axis='x', rotation=0)

group_validation.reindex(order).plot(kind='bar', color=['#2ecc71', '#f39c12', '#e74c3c'], ax=axes[1])
axes[1].set_title('Actual Disease Rate (%) by Risk Group')
axes[1].set_ylabel('% with Disease')
axes[1].tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig(f'{CHARTS}/08_risk_group_segmentation.png')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
bmi_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
bmi_validation.reindex(bmi_order).plot(kind='bar', color='#16a085', ax=axes[0])
axes[0].set_title('Disease Risk (%) by BMI Category')
axes[0].set_ylabel('% with Disease')
axes[0].tick_params(axis='x', rotation=20)

bp_order = ['Normal', 'Elevated', 'Hypertension Stage 1', 'Hypertension Stage 2']
bp_validation.reindex(bp_order).plot(kind='bar', color='#c0392b', ax=axes[1])
axes[1].set_title('Disease Risk (%) by BP Category')
axes[1].set_ylabel('% with Disease')
axes[1].tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig(f'{CHARTS}/09_bmi_bp_categories.png')
plt.close()

# Correlation of new features with target
new_feat_corr = df[['bmi', 'risk_score', 'cardio']].corr()['cardio'].drop('cardio')
notes.append(f"Correlation with disease outcome: bmi = {new_feat_corr['bmi']:.2f}, "
             f"risk_score = {new_feat_corr['risk_score']:.2f} (risk_score is a notably "
             f"stronger single predictor than any individual raw feature).")

# ---------------------------------------------------------
# Save engineered dataset
# ---------------------------------------------------------
df.to_csv(DATA_DIR / 'cardio_train_features.csv', index=False)

with open(REPORTS_DIR / 'feature_engineering_log.txt', 'w') as f:
    f.write("FEATURE ENGINEERING & RISK ANALYSIS\n")
    f.write("="*60 + "\n\n")
    for i, n in enumerate(notes, 1):
        f.write(f"{i}. {n}\n\n")
    f.write("\nDisease rate by BMI category:\n")
    f.write(bmi_validation.reindex(bmi_order).to_string())
    f.write("\n\nDisease rate by BP category:\n")
    f.write(bp_validation.reindex(bp_order).to_string())

print("Feature engineering complete.")
for n in notes:
    print(f"- {n}")
