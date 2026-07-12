"""
Data Preprocessing Script
Project: Employee Attrition Prediction & HR Analytics

Loads the raw IBM HR Analytics Employee Attrition dataset, explores it,
cleans it, and saves a processed version for downstream EDA and modeling.
"""

import pandas as pd
import numpy as np
import os

# Anchor all paths to this script's location so the script works the same way
# whether it's run from a terminal, VS Code's "Run" button, or another machine.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_attrition_data.csv")


def load_data(path):
    df = pd.read_csv(path)
    return df


def explore_data(df):
    print("Shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nSummary statistics:\n", df.describe())


def clean_data(df):
    df = df.copy()

    # Drop duplicates if any
    df.drop_duplicates(inplace=True)

    # Drop constant / non-informative columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    df.drop(columns=constant_cols, inplace=True, errors="ignore")

    # EmployeeNumber is an identifier, not a predictive feature
    if "EmployeeNumber" in df.columns:
        df.drop(columns=["EmployeeNumber"], inplace=True)

    # Confirm no missing values remain
    df.dropna(inplace=True)

    return df


def main():
    df = load_data(RAW_DATA_PATH)
    explore_data(df)

    cleaned_df = clean_data(df)

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\nCleaned dataset saved to {PROCESSED_DATA_PATH}")
    print("Final shape:", cleaned_df.shape)


if __name__ == "__main__":
    main()
