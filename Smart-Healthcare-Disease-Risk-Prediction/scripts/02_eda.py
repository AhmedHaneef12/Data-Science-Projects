"""
Task 04 - Part 2: Exploratory Data Analysis (EDA)
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

findings = []

# ---------------------------------------------------------
# 1. Risk distribution (target variable)
# ---------------------------------------------------------
risk_counts = df['cardio'].value_counts(normalize=True) * 100
findings.append(f"Disease prevalence: {risk_counts[1]:.1f}% of patients have cardiovascular disease "
                 f"(cardio=1), {risk_counts[0]:.1f}% do not. Dataset is well-balanced for modeling.")

fig, ax = plt.subplots(figsize=(6, 5))
sns.countplot(data=df, x='cardio', hue='cardio', palette=['#2ecc71', '#e74c3c'], legend=False, ax=ax)
ax.set_xticks([0, 1])
ax.set_xticklabels(['No Disease', 'Disease'])
ax.set_title('Disease Risk Distribution', fontsize=13, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('Number of Patients')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width()/2, p.get_height()),
                ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(f'{CHARTS}/01_risk_distribution.png')
plt.close()

# ---------------------------------------------------------
# 2. Age vs disease risk
# ---------------------------------------------------------
df['age_group'] = pd.cut(df['age_years'], bins=[29, 40, 45, 50, 55, 60, 65],
                          labels=['30-40', '41-45', '46-50', '51-55', '56-60', '61-65'])
age_risk = df.groupby('age_group', observed=True)['cardio'].mean() * 100
findings.append(f"Disease risk rises sharply with age: from {age_risk.iloc[0]:.1f}% in the 30-40 group "
                 f"to {age_risk.iloc[-1]:.1f}% in the 61-65 group.")

fig, ax = plt.subplots(figsize=(7, 5))
age_risk.plot(kind='bar', color='#3498db', ax=ax)
ax.set_title('Disease Risk (%) by Age Group', fontsize=13, fontweight='bold')
ax.set_xlabel('Age Group (years)')
ax.set_ylabel('% with Disease')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{CHARTS}/02_age_vs_risk.png')
plt.close()

# ---------------------------------------------------------
# 3. Blood pressure analysis (box plots by disease status)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
sns.boxplot(data=df, x='cardio', y='ap_hi', hue='cardio', palette=['#2ecc71', '#e74c3c'], legend=False, ax=axes[0])
axes[0].set_xticklabels(['No Disease', 'Disease'])
axes[0].set_title('Systolic BP by Disease Status')
axes[0].set_xlabel('')
sns.boxplot(data=df, x='cardio', y='ap_lo', hue='cardio', palette=['#2ecc71', '#e74c3c'], legend=False, ax=axes[1])
axes[1].set_xticklabels(['No Disease', 'Disease'])
axes[1].set_title('Diastolic BP by Disease Status')
axes[1].set_xlabel('')
plt.tight_layout()
plt.savefig(f'{CHARTS}/03_bp_boxplots.png')
plt.close()

ap_hi_diff = df.groupby('cardio')['ap_hi'].mean()
ap_lo_diff = df.groupby('cardio')['ap_lo'].mean()
findings.append(f"Patients with disease show notably higher blood pressure: average systolic "
                 f"{ap_hi_diff[1]:.0f} mmHg vs {ap_hi_diff[0]:.0f} mmHg in healthy patients; "
                 f"diastolic {ap_lo_diff[1]:.0f} vs {ap_lo_diff[0]:.0f} mmHg.")

# ---------------------------------------------------------
# 4. Cholesterol & Glucose impact
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
chol_risk = df.groupby('cholesterol')['cardio'].mean() * 100
chol_risk.index = ['Normal', 'Above Normal', 'Well Above Normal']
chol_risk.plot(kind='bar', color='#e67e22', ax=axes[0])
axes[0].set_title('Disease Risk (%) by Cholesterol Level')
axes[0].set_ylabel('% with Disease')
axes[0].tick_params(axis='x', rotation=20)

gluc_risk = df.groupby('gluc')['cardio'].mean() * 100
gluc_risk.index = ['Normal', 'Above Normal', 'Well Above Normal']
gluc_risk.plot(kind='bar', color='#9b59b6', ax=axes[1])
axes[1].set_title('Disease Risk (%) by Glucose Level')
axes[1].set_ylabel('% with Disease')
axes[1].tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig(f'{CHARTS}/04_cholesterol_glucose_risk.png')
plt.close()

findings.append(f"Cholesterol is strongly linked to risk: {chol_risk.iloc[0]:.1f}% disease rate at normal "
                 f"levels vs {chol_risk.iloc[2]:.1f}% at well-above-normal levels.")
findings.append(f"Glucose shows a similar but weaker pattern: {gluc_risk.iloc[0]:.1f}% at normal levels "
                 f"vs {gluc_risk.iloc[2]:.1f}% at well-above-normal levels.")

# ---------------------------------------------------------
# 5. Lifestyle factors (smoking, alcohol, activity)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, col, label, colors in zip(
        axes, ['smoke', 'alco', 'active'],
        ['Smoking', 'Alcohol Intake', 'Physical Activity'],
        [['#95a5a6', '#e74c3c'], ['#95a5a6', '#e74c3c'], ['#e74c3c', '#2ecc71']]):
    rates = df.groupby(col)['cardio'].mean() * 100
    rates.index = ['No', 'Yes']
    rates.plot(kind='bar', color=colors, ax=ax)
    ax.set_title(f'Risk by {label}')
    ax.set_ylabel('% with Disease')
    ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig(f'{CHARTS}/05_lifestyle_factors.png')
plt.close()

smoke_risk = df.groupby('smoke')['cardio'].mean() * 100
active_risk = df.groupby('active')['cardio'].mean() * 100
findings.append(f"Lifestyle factors show smaller individual effects: smokers have a {smoke_risk[1]:.1f}% "
                 f"disease rate vs {smoke_risk[0]:.1f}% for non-smokers (a modest, somewhat counterintuitive "
                 f"gap, likely confounded by age and other factors in self-reported data).")
findings.append(f"Physically active patients have a {active_risk[1]:.1f}% disease rate vs "
                 f"{active_risk[0]:.1f}% for inactive patients, a small protective effect.")

# ---------------------------------------------------------
# 6. Correlation heatmap
# ---------------------------------------------------------
corr_cols = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
             'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            cbar_kws={'label': 'Correlation'})
ax.set_title('Correlation Heatmap - Health Indicators', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{CHARTS}/06_correlation_heatmap.png')
plt.close()

top_corr = corr['cardio'].drop('cardio').sort_values(key=abs, ascending=False)
findings.append("Top correlates with disease risk: " +
                 ", ".join([f"{idx} ({val:.2f})" for idx, val in top_corr.head(5).items()]) + ".")

# ---------------------------------------------------------
# 7. Weight/Height distributions
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.histplot(df['height'], bins=40, color='#3498db', ax=axes[0], kde=True)
axes[0].set_title('Height Distribution (cm)')
sns.histplot(df['weight'], bins=40, color='#e67e22', ax=axes[1], kde=True)
axes[1].set_title('Weight Distribution (kg)')
plt.tight_layout()
plt.savefig(f'{CHARTS}/07_height_weight_distribution.png')
plt.close()

# ---------------------------------------------------------
# Save findings
# ---------------------------------------------------------
stats_summary = df[['age_years', 'height', 'weight', 'ap_hi', 'ap_lo']].describe().T

with open(REPORTS_DIR / 'eda_findings.txt', 'w') as f:
    f.write("EXPLORATORY DATA ANALYSIS - KEY FINDINGS\n")
    f.write("="*60 + "\n\n")
    for i, finding in enumerate(findings, 1):
        f.write(f"{i}. {finding}\n\n")
    f.write("\nSTATISTICAL SUMMARY (numeric health indicators)\n")
    f.write("="*60 + "\n")
    f.write(stats_summary.to_string())

print("EDA complete. Findings:")
for f_ in findings:
    print(f"- {f_}")
print("\nCharts saved to /home/claude/charts/")
