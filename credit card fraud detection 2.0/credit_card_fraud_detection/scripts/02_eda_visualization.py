"""
02_eda_visualization.py

Exploratory Data Analysis (EDA)
--------------------------------
Generates the visual analysis used to understand class imbalance, transaction
amount behavior, feature correlation and time-based fraud patterns.

All charts are written as PNG files to the `charts/` directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Path setup (file-anchored, portable across machines)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHARTS_DIR = os.path.join(PROJECT_ROOT, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

CLEANED_PATH = os.path.join(DATA_DIR, "cleaned_transactions.csv")

sns.set_style("whitegrid")
PALETTE = {"legit": "#2E86AB", "fraud": "#E63946"}


def save(fig, name):
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_class_distribution(df):
    counts = df["Class"].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(["Legitimate", "Fraud"], counts.values,
                color=[PALETTE["legit"], PALETTE["fraud"]])
    axes[0].set_title("Transaction Class Distribution (Counts)")
    axes[0].set_ylabel("Number of Transactions")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")

    axes[1].pie(counts.values, labels=["Legitimate", "Fraud"], autopct="%1.2f%%",
                colors=[PALETTE["legit"], PALETTE["fraud"]], startangle=90)
    axes[1].set_title("Transaction Class Distribution (%)")

    fig.suptitle("Class Imbalance in Transaction Data", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "01_class_distribution.png")


def plot_amount_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for label, name, color in [(0, "Legitimate", PALETTE["legit"]), (1, "Fraud", PALETTE["fraud"])]:
        subset = df.loc[df["Class"] == label, "Amount"]
        axes[0].hist(subset.clip(upper=subset.quantile(0.99)), bins=40, alpha=0.6,
                     label=name, color=color, density=True)
    axes[0].set_title("Transaction Amount Distribution")
    axes[0].set_xlabel("Amount")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    sns.boxplot(data=df, x="Class", y="Amount", ax=axes[1],
                palette=[PALETTE["legit"], PALETTE["fraud"]])
    axes[1].set_yscale("log")
    axes[1].set_xticklabels(["Legitimate", "Fraud"])
    axes[1].set_title("Amount by Class (log scale)")

    fig.suptitle("Transaction Amount Patterns", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "02_amount_distribution.png")


def plot_time_patterns(df):
    df = df.copy()
    df["Hour"] = (df["Time"] % (24 * 3600)) // 3600

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fraud_by_hour = df[df["Class"] == 1].groupby("Hour").size()
    legit_by_hour = df[df["Class"] == 0].groupby("Hour").size()
    legit_rate = (legit_by_hour / legit_by_hour.sum() * 100).reindex(range(24), fill_value=0)
    fraud_rate = (fraud_by_hour / fraud_by_hour.sum() * 100).reindex(range(24), fill_value=0)

    x = np.arange(24)
    width = 0.4
    ax.bar(x - width / 2, legit_rate.values, width, label="Legitimate", color=PALETTE["legit"])
    ax.bar(x + width / 2, fraud_rate.values, width, label="Fraud", color=PALETTE["fraud"])
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Share of Transactions (%)")
    ax.set_title("Transaction Timing Patterns by Hour of Day")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    save(fig, "03_time_patterns.png")


def plot_correlation_heatmap(df):
    corr_cols = [c for c in df.columns if c.startswith("V")][:12] + ["Amount", "Class"]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=False, ax=ax,
                cbar_kws={"label": "Correlation"})
    ax.set_title("Feature Correlation Heatmap (subset of V-features)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "04_correlation_heatmap.png")


def plot_feature_correlation_with_class(df):
    v_cols = [c for c in df.columns if c.startswith("V")]
    corr_with_class = df[v_cols + ["Class"]].corr()["Class"].drop("Class").sort_values()

    fig, ax = plt.subplots(figsize=(8, 9))
    colors = [PALETTE["fraud"] if v > 0 else PALETTE["legit"] for v in corr_with_class.values]
    ax.barh(corr_with_class.index, corr_with_class.values, color=colors)
    ax.set_title("Feature Correlation with Fraud Label", fontsize=13, fontweight="bold")
    ax.set_xlabel("Correlation with Class")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    save(fig, "05_feature_correlation_with_class.png")


def main():
    print("=" * 60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(CLEANED_PATH)
    print(f"Loaded cleaned dataset: {df.shape}")

    plot_class_distribution(df)
    plot_amount_distribution(df)
    plot_time_patterns(df)
    plot_correlation_heatmap(df)
    plot_feature_correlation_with_class(df)

    print("\nEDA complete. All charts saved to the charts/ directory.")


if __name__ == "__main__":
    main()
