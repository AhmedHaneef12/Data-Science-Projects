"""
01_data_preparation_eda.py
---------------------------
Data Science Task #07 - Credit Card Fraud Detection
Step 1: Data Preparation & Exploratory Data Analysis (EDA)

Original dataset source (full version, ~150 MB, 284,807 transactions):
Kaggle - Credit Card Fraud Detection
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

NOTE: The original Kaggle file was too large to open/process in this
environment, so a reduced random sample of 1,000 transactions
(creditcard_small_1000.csv) was supplied and used for this project instead.
All steps below follow the same methodology that would be used on the
full dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

RAW_PATH = "data/creditcard_small_1000_raw.csv"
CLEAN_PATH = "data/creditcard_cleaned.csv"
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------
df = pd.read_csv(RAW_PATH)
print("Dataset shape:", df.shape)

# ---------------------------------------------------------------
# 2. Explore dataset structure
# ---------------------------------------------------------------
print("\nColumn list:", list(df.columns))
print("\nFirst 5 rows:\n", df.head())

# ---------------------------------------------------------------
# 3. Data types & summary statistics
# ---------------------------------------------------------------
print("\nData types:\n", df.dtypes)
print("\nSummary statistics:\n", df.describe().T)

# ---------------------------------------------------------------
# 4. Missing values & duplicate records
# ---------------------------------------------------------------
missing = df.isnull().sum()
n_missing = missing.sum()
n_duplicates = df.duplicated().sum()
print(f"\nTotal missing values: {n_missing}")
print(f"Total duplicate rows: {n_duplicates}")

# ---------------------------------------------------------------
# 5. Clean the dataset
# ---------------------------------------------------------------
df_clean = df.copy()
if n_missing > 0:
    df_clean = df_clean.dropna()
if n_duplicates > 0:
    df_clean = df_clean.drop_duplicates()

# Sanity-check Class column only contains 0/1
df_clean["Class"] = df_clean["Class"].astype(int)
assert set(df_clean["Class"].unique()).issubset({0, 1})

df_clean.to_csv(CLEAN_PATH, index=False)
print(f"\nCleaned dataset saved to {CLEAN_PATH} -> shape {df_clean.shape}")

# ---------------------------------------------------------------
# 6. Class distribution (fraud vs non-fraud)
# ---------------------------------------------------------------
class_counts = df_clean["Class"].value_counts().sort_index()
fraud_pct = class_counts.get(1, 0) / len(df_clean) * 100
print("\nClass distribution:\n", class_counts)
print(f"Fraud percentage: {fraud_pct:.4f}%")

plt.figure(figsize=(5, 4))
ax = sns.countplot(x="Class", data=df_clean, hue="Class",
                    palette={0: "#2E75B6", 1: "#D9534F"}, legend=False)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Non-Fraud (0)", "Fraud (1)"])
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=10)
plt.title("Class Distribution: Fraud vs Non-Fraud Transactions")
plt.ylabel("Number of Transactions")
plt.xlabel("")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/01_class_distribution.png")
plt.close()

# ---------------------------------------------------------------
# 7. Transaction amount distribution
# ---------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.histplot(df_clean["Amount"], bins=40, color="#2E75B6", kde=True)
plt.title("Distribution of Transaction Amount")
plt.xlabel("Amount ($)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/02_amount_distribution.png")
plt.close()

# ---------------------------------------------------------------
# 8. Amount vs Class (boxplot) - relationship between amount & fraud
# ---------------------------------------------------------------
plt.figure(figsize=(5, 4))
sns.boxplot(x="Class", y="Amount", data=df_clean, hue="Class",
            palette={0: "#2E75B6", 1: "#D9534F"}, legend=False)
plt.xticks([0, 1], ["Non-Fraud", "Fraud"])
plt.title("Transaction Amount by Class")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/03_amount_by_class.png")
plt.close()

# ---------------------------------------------------------------
# 9. Transaction Time distribution
# ---------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.histplot(df_clean["Time"], bins=40, color="#5B9BD5")
plt.title("Distribution of Transaction Time")
plt.xlabel("Time (seconds since first transaction)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/04_time_distribution.png")
plt.close()

# ---------------------------------------------------------------
# 10. Correlation heatmap (V1-V28, Amount, Class)
# ---------------------------------------------------------------
plt.figure(figsize=(12, 10))
corr = df_clean.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
plt.title("Correlation Heatmap of All Features")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/05_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------------
# 11. Correlation of each feature with Class (fraud)
# ---------------------------------------------------------------
class_corr = corr["Class"].drop("Class").sort_values()
plt.figure(figsize=(7, 8))
class_corr.plot(kind="barh", color=np.where(class_corr > 0, "#D9534F", "#2E75B6"))
plt.title("Feature Correlation with Fraud (Class)")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/06_feature_correlation_with_class.png")
plt.close()

print("\nAll EDA charts saved to the 'charts/' folder.")
print("Step 1 (Data Preparation & EDA) complete.")
