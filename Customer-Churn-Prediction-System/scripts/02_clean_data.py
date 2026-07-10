import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "telco_churn_raw.csv"
OUT_PATH = BASE_DIR / "data" / "processed" / "telco_churn_cleaned.csv"

df = pd.read_csv(RAW_PATH)
print("Raw shape:", df.shape)

dupes = df.duplicated().sum()
print("Duplicate rows:", dupes)
df = df.drop_duplicates().reset_index(drop=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Missing TotalCharges before fill:", df["TotalCharges"].isna().sum())
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("Missing MultipleLines before fill:", df["MultipleLines"].isna().sum())
mode_val = df["MultipleLines"].mode()[0]
df["MultipleLines"] = df["MultipleLines"].fillna(mode_val)

print("Any remaining nulls?\n", df.isna().sum()[df.isna().sum() > 0])
print("Cleaned shape:", df.shape)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print("Saved cleaned dataset to", OUT_PATH)
