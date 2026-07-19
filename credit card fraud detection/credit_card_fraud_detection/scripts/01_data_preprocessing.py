"""Data Cleaning & Preprocessing."""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RAW_PATH = os.path.join(DATA_DIR, "raw_transactions.csv")
CLEANED_PATH = os.path.join(DATA_DIR, "cleaned_transactions.csv")

RANDOM_STATE = 42


def generate_raw_dataset(n_normal=6000, n_fraud=90, random_state=RANDOM_STATE):
    """Builds a transaction dataset with Time, V1-V28, Amount and Class columns."""
    rng = np.random.default_rng(random_state)
    n_total = n_normal + n_fraud

    time_seconds = np.sort(rng.uniform(0, 2 * 24 * 3600, n_total))
    v_features = rng.normal(loc=0.0, scale=1.0, size=(n_total, 28))

    labels = np.array([0] * n_normal + [1] * n_fraud)
    shuffle_idx = rng.permutation(n_total)
    labels = labels[shuffle_idx]
    fraud_mask = labels == 1

    # Shift a subset of features for fraud cases so the signal is realistic
    # (partial and noisy) rather than perfectly separable.
    informative_cols = [3, 9, 13, 16, 21]
    strong_signal_mask = fraud_mask & (rng.random(n_total) < 0.9)
    for col in informative_cols:
        shift = rng.normal(loc=3.0, scale=1.0, size=n_total)
        v_features[strong_signal_mask, col] += shift[strong_signal_mask] * rng.choice(
            [-1, 1], size=strong_signal_mask.sum()
        )

    # Legitimate amounts follow a right-skewed distribution; fraud skews
    # toward small "testing" charges with occasional large outliers.
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

    # Inject duplicates and missing values so cleaning has real work to do.
    dup_idx = rng.choice(df.index, size=25, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    missing_idx = rng.choice(df.index, size=15, replace=False)
    df.loc[missing_idx, "Amount"] = np.nan
    missing_idx_v = rng.choice(df.index, size=10, replace=False)
    df.loc[missing_idx_v, "V7"] = np.nan

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def clean_dataset(df):
    print(f"Initial shape: {df.shape}")
    print(f"Missing values before cleaning:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    n_before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {n_before - len(df)} duplicate rows")

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    n_before = len(df)
    df = df[df["Amount"] >= 0]
    print(f"Removed {n_before - len(df)} rows with invalid Amount")

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
