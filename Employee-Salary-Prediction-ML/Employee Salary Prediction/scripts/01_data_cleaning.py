"""
Task #06 - Employee Salary Prediction
Script 01: Data Understanding & Cleaning
Dataset : Employee Salary Dataset — Kaggle
Source  : https://www.kaggle.com/datasets/prince7489/employee-salary-dataset/data
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent
RAW  = BASE / 'data' / 'employee_salary_dataset.csv'
CLEAN= BASE / 'data' / 'cleaned_employee_salary.csv'

print("=" * 60)
print("  TASK #06 — DATA UNDERSTANDING & CLEANING")
print("=" * 60)
print(f"\n  Source : Kaggle — Employee Salary Dataset")
print(f"  URL    : https://www.kaggle.com/datasets/prince7489/employee-salary-dataset/data")

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW)
print(f"\n[1] Dataset Loaded")
print(f"    Shape   : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"    Columns : {list(df.columns)}")
print(f"\n    First 5 rows:\n{df.head().to_string()}")

# ── 2. Structure Exploration ─────────────────────────────────────────────────
print(f"\n[2] Data Types")
print(df.dtypes.to_string())

print(f"\n[3] Descriptive Statistics")
print(df.describe(include='all').to_string())

# ── 3. Missing Values ─────────────────────────────────────────────────────────
print(f"\n[4] Missing Values")
missing = df.isnull().sum()
print(missing.to_string())
print(f"    Total missing: {missing.sum()}")

# ── 4. Duplicates ─────────────────────────────────────────────────────────────
print(f"\n[5] Duplicate Records: {df.duplicated().sum()}")

# ── 5. Unique Values per Categorical ─────────────────────────────────────────
print(f"\n[6] Categorical Value Counts")
for col in ['Department', 'Education_Level', 'Gender', 'City']:
    print(f"\n    {col}:")
    print(df[col].value_counts().to_string())

# ── 6. Outlier Check on Monthly_Salary ───────────────────────────────────────
Q1  = df['Monthly_Salary'].quantile(0.25)
Q3  = df['Monthly_Salary'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df['Monthly_Salary'] < lower) | (df['Monthly_Salary'] > upper)]
print(f"\n[7] Salary Outlier Check (1.5×IQR)")
print(f"    Lower fence : ₹{lower:,.0f}  |  Upper fence : ₹{upper:,.0f}")
print(f"    Outliers found: {len(outliers)}")
if len(outliers) > 0:
    print(outliers[['EmployeeID', 'Department', 'Experience_Years', 'Monthly_Salary']].to_string())

# ── 7. Begin Cleaning ─────────────────────────────────────────────────────────
df_clean = df.copy()

# Drop Name column — personal identifier, not useful for ML
df_clean = df_clean.drop(columns=['Name'])
print(f"\n[8] 'Name' column dropped (PII — not relevant for prediction)")

# Standardise Education_Level labels (consistent casing)
edu_std = {'High School': 'High School', 'Bachelor': 'Bachelor', 'Master': 'Master', 'PhD': 'PhD'}
df_clean['Education_Level'] = df_clean['Education_Level'].map(edu_std)
print(f"    Education levels standardised: {sorted(df_clean['Education_Level'].unique())}")

# No missing values or duplicates to handle — dataset is already clean
print(f"\n[9] Post-Cleaning Validation")
print(f"    Missing values : {df_clean.isnull().sum().sum()}")
print(f"    Duplicate rows : {df_clean.duplicated().sum()}")

# ── 8. Encode Categorical Columns ─────────────────────────────────────────────
# Education: ordinal
edu_order = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
df_clean['education_encoded'] = df_clean['Education_Level'].map(edu_order)

# Gender: binary
df_clean['gender_encoded'] = (df_clean['Gender'] == 'Male').astype(int)

# Department: label encode
dept_list = sorted(df_clean['Department'].unique())
dept_map  = {d: i for i, d in enumerate(dept_list)}
df_clean['department_encoded'] = df_clean['Department'].map(dept_map)

# City: label encode
city_list = sorted(df_clean['City'].unique())
city_map  = {c: i for i, c in enumerate(city_list)}
df_clean['city_encoded'] = df_clean['City'].map(city_map)

print(f"\n[10] Encoding Applied")
print(f"     education_encoded  : {edu_order}")
print(f"     gender_encoded     : Male=1, Female=0")
print(f"     department_encoded : {dept_map}")
print(f"     city_encoded       : {city_map}")

# ── 9. Final Summary ──────────────────────────────────────────────────────────
print(f"\n[11] Cleaned Dataset Summary")
print(f"     Shape          : {df_clean.shape}")
print(f"     Monthly Salary : ₹{df_clean['Monthly_Salary'].min():,.0f} – ₹{df_clean['Monthly_Salary'].max():,.0f}")
print(f"     Mean Salary    : ₹{df_clean['Monthly_Salary'].mean():,.0f}")
print(f"     Median Salary  : ₹{df_clean['Monthly_Salary'].median():,.0f}")
print(f"\n    Columns:\n{df_clean.dtypes.to_string()}")

df_clean.to_csv(CLEAN, index=False)
print(f"\n✓ Cleaned dataset saved → {CLEAN}")
print("=" * 60)
