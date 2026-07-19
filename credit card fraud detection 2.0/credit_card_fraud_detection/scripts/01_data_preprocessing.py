"""
01_data_preprocessing.py

Data Cleaning & Preprocessing
------------------------------
This script builds the working transaction dataset used throughout the project
and performs the initial cleaning and preprocessing steps:

    1. Load the raw transaction data
    2. Inspect structure, data types and missing values
    3. Remove duplicate records
    4. Validate value ranges (Amount, Time)
    5. Save a cleaned dataset for downstream use

All paths are anchored to this file's location so the script can be run from
any working directory or machine.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup (file-anchored, portable across machines)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RAW_PATH = os.path.join(DATA_DIR, "raw_transactions.csv")
CLEANED_PATH = os.path.join(DATA_DIR, "cleaned_transactions.csv")

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Step 1: Build the raw transaction dataset
# ---------------------------------------------------------------------------
def generate_raw_dataset(n_normal=6000, n_fraud=90, random_state=RANDOM_STATE):
    """
    Builds a transaction-level dataset that mirrors the structure commonly
    used for credit card fraud studies: a `Time` column (seconds elapsed),
    28 anonymized/PCA-style numeric features (V1-V28), a transaction
    `Amount`, and a binary `Class` label (1 = fraud, 0 = legitimate).

    Fraudulent transactions are modeled with different feature and amount
    distributions than legitimate ones, and a small amount of noise,
    duplication and missing data is injected so the cleaning steps below
    have real work to do.
    """
    rng = np.random.default_rng(random_state)
    n_total = n_normal + n_fraud

    # Chronological "Time" field: seconds elapsed across a 2-day window
    time_seconds = np.sort(rng.uniform(0, 2 * 24 * 3600, n_total))

    # 28 anonymized numeric features (V1-V28), standard-normal for legitimate
    # transactions and shifted/scaled for fraudulent ones, to emulate the
    # separation seen in real anonymized fraud data.
    v_features = rng.normal(loc=0.0, scale=1.0, size=(n_total, 28))

    labels = np.array([0] * n_normal + [1] * n_fraud)
    shuffle_idx = rng.permutation(n_total)
    labels = labels[shuffle_idx]

    fraud_mask = labels == 1
    n_fraud_actual = fraud_mask.sum()

    # Shift a handful of informative features for fraud cases so later
    # feature-importance analysis has genuine signal to recover. Only a
    # portion of fraud cases carry a strong signal, and the shift is modest,
    # so the classes overlap the way real anonymized fraud features do
    # rather than separating perfectly.
    informative_cols = [3, 9, 13, 16, 21]
    strong_signal_mask = fraud_mask & (rng.random(n_total) < 0.9)
    for col in informative_cols:
        shift = rng.normal(loc=3.0, scale=1.0, size=n_total)
        v_features[strong_signal_mask, col] += shift[strong_signal_mask] * rng.choice(
            [-1, 1], size=strong_signal_mask.sum()
        )

    # Transaction amount: legitimate purchases follow a right-skewed
    # distribution; fraudulent ones skew toward smaller "testing" charges
    # with occasional large outliers.
    amount = np.where(
        labels == 0,
        rng.gamma(shape=1.4, scale=45, size=n_total),
        np.where(
            rng.random(n_total) < 0.75,
            rng.gamma(shape=1.0, scale=12, size=n_total),
            rng.gamma(shape=2.0, scale=250, size=n_total),
        ),
    )
    amount = np.round(amount, 2)

    df = pd.DataFrame(v_features, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.round(time_seconds, 0))
    df["Amount"] = amount
    df["Class"] = labels

    # --- Inject realistic data-quality issues for the cleaning step ---
    # Duplicate a small number of rows
    dup_idx = rng.choice(df.index, size=25, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # Introduce a few missing values in Amount and a couple of V columns
    missing_idx = rng.choice(df.index, size=15, replace=False)
    df.loc[missing_idx, "Amount"] = np.nan
    missing_idx_v = rng.choice(df.index, size=10, replace=False)
    df.loc[missing_idx_v, "V7"] = np.nan

    # Shuffle row order so Time is not trivially pre-sorted for the learner
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Step 2: Cleaning routine
# ---------------------------------------------------------------------------
def clean_dataset(df):
    print(f"Initial shape: {df.shape}")
    print(f"Missing values before cleaning:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Remove exact duplicate rows
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {n_before - len(df)} duplicate rows")

    # Impute missing numeric values with the column median (robust to skew)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Drop any transactions with a non-positive amount (data entry errors)
    n_before = len(df)
    df = df[df["Amount"] >= 0]
    print(f"Removed {n_before - len(df)} rows with invalid Amount")

    # Ensure correct dtypes
    df["Class"] = df["Class"].astype(int)
    df["Time"] = df["Time"].astype(float)

    df = df.reset_index(drop=True)
    print(f"Final shape: {df.shape}")
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

    return df


def main():
    print("=" * 60)
    print("STEP 1: DATA CLEANING & PREPROCESSING")
    print("=" * 60)

    df_raw = generate_raw_dataset()
    df_raw.to_csv(RAW_PATH, index=False)
    print(f"\nRaw dataset saved to: {RAW_PATH}")

    df_clean = clean_dataset(df_raw)
    df_clean.to_csv(CLEANED_PATH, index=False)
    print(f"\nCleaned dataset saved to: {CLEANED_PATH}")

    fraud_rate = df_clean["Class"].mean() * 100
    print(f"\nClass distribution:")
    print(df_clean["Class"].value_counts())
    print(f"Fraud rate: {fraud_rate:.3f}%")


if __name__ == "__main__":
    main()
