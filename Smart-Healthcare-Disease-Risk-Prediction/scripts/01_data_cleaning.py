"""
Task 04 - Part 1: Data Understanding & Cleaning
Smart Healthcare Data Analysis & Disease Risk Prediction
Cardiovascular Disease Dataset (Kaggle - sulianova)
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Folder structure: project_root/{data, scripts, charts, reports}
# This makes the script work no matter where the project folder is on your laptop.
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / 'data'
REPORTS_DIR = BASE / 'reports'
REPORTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv(DATA_DIR / 'cardio_train_raw.csv', sep=';')

print("="*60)
print("DATASET OVERVIEW")
print("="*60)
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nColumn dtypes:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())

print("\n" + "="*60)
print("MISSING VALUES ANALYSIS")
print("="*60)
missing = df.isnull().sum()
print(missing)
print(f"\nTotal missing cells: {missing.sum()}")

print("\n" + "="*60)
print("DUPLICATE RECORDS")
print("="*60)
dupes = df.duplicated().sum()
print(f"Fully duplicated rows: {dupes}")
dupes_no_id = df.drop(columns=['id']).duplicated().sum()
print(f"Duplicate rows (ignoring id): {dupes_no_id}")

print("\n" + "="*60)
print("STATISTICAL SUMMARY (RAW)")
print("="*60)
print(df.describe().T)

# ---------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------
notes = []
df_clean = df.copy()

# --- Duplicates ---
before = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=df_clean.columns.difference(['id']))
notes.append(f"Removed {before - len(df_clean)} duplicate records (ignoring id).")

# --- Age: convert days -> years (sanity check + new readable column) ---
df_clean['age_years'] = (df_clean['age'] / 365.25).round(1)
notes.append("Converted 'age' (days) into a readable 'age_years' column for analysis; raw 'age' kept intact.")
age_min, age_max = df_clean['age_years'].min(), df_clean['age_years'].max()
notes.append(f"Age range after conversion: {age_min:.1f} - {age_max:.1f} years (clinically plausible, no cleaning needed).")

# --- Height: realistic adult range 120-220 cm ---
before = len(df_clean)
df_clean = df_clean[(df_clean['height'] >= 120) & (df_clean['height'] <= 220)]
notes.append(f"Removed {before - len(df_clean)} records with implausible height (<120cm or >220cm).")

# --- Weight: realistic adult range 30-200 kg ---
before = len(df_clean)
df_clean = df_clean[(df_clean['weight'] >= 30) & (df_clean['weight'] <= 200)]
notes.append(f"Removed {before - len(df_clean)} records with implausible weight (<30kg or >200kg).")

# --- Blood Pressure: ap_hi (systolic) and ap_lo (diastolic) ---
# Known issue in this dataset: negative values, zeros, and values in the thousands (data entry errors)
# Clinically plausible ranges used: systolic 70-250 mmHg, diastolic 40-200 mmHg
before = len(df_clean)
df_clean = df_clean[(df_clean['ap_hi'] >= 70) & (df_clean['ap_hi'] <= 250)]
df_clean = df_clean[(df_clean['ap_lo'] >= 40) & (df_clean['ap_lo'] <= 200)]
notes.append(f"Removed {before - len(df_clean)} records with implausible blood pressure readings "
             f"(systolic outside 70-250 mmHg or diastolic outside 40-200 mmHg).")

# --- Logical check: systolic must be >= diastolic ---
before = len(df_clean)
df_clean = df_clean[df_clean['ap_hi'] >= df_clean['ap_lo']]
notes.append(f"Removed {before - len(df_clean)} records where diastolic (ap_lo) exceeded systolic (ap_hi), "
             f"a physiological impossibility.")

# --- Categorical sanity checks ---
for col in ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']:
    vals = sorted(df_clean[col].unique())
    notes.append(f"Column '{col}' contains valid categorical values: {vals}")

# --- Data types ---
cat_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
for c in cat_cols:
    df_clean[c] = df_clean[c].astype('int8')
notes.append("Cast categorical/binary columns to compact integer dtype.")

removed_total = len(df) - len(df_clean)
pct_removed = removed_total / len(df) * 100

print("\n" + "="*60)
print("CLEANING STRATEGY LOG")
print("="*60)
for n in notes:
    print(f"- {n}")
print(f"\nTOTAL rows removed: {removed_total} ({pct_removed:.2f}% of original {len(df)})")
print(f"Final cleaned dataset: {len(df_clean)} rows x {df_clean.shape[1]} columns")

# ---------------------------------------------------------
# 3. SAVE CLEANED DATASET
# ---------------------------------------------------------
df_clean.to_csv(DATA_DIR / 'cardio_train_cleaned.csv', index=False)
print("\nSaved cleaned dataset to cardio_train_cleaned.csv")

# Save the cleaning log to a text file for the report
with open(REPORTS_DIR / 'cleaning_log.txt', 'w') as f:
    f.write("DATA CLEANING STRATEGY LOG\n")
    f.write("="*60 + "\n\n")
    f.write(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")
    f.write(f"Missing values found: {missing.sum()}\n")
    f.write(f"Fully duplicated rows: {dupes}\n\n")
    f.write("Cleaning steps applied:\n")
    for n in notes:
        f.write(f"- {n}\n")
    f.write(f"\nTotal rows removed: {removed_total} ({pct_removed:.2f}%)\n")
    f.write(f"Final cleaned dataset: {len(df_clean)} rows x {df_clean.shape[1]} columns\n")

print("\nDone.")
