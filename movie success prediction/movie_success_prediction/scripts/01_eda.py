"""
Exploratory Data Analysis (EDA) Script
----------------------------------------
Loads the raw movie dataset, cleans it (missing values, duplicates,
encoding groundwork), and produces a set of visualizations that
highlight trends relevant to movie commercial success.

Outputs:
    data/movies_dataset_clean.csv
    charts/*.png
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# File-anchored paths
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

RAW_PATH = os.path.join(DATA_DIR, "movies_dataset.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "movies_dataset_clean.csv")

sns.set_style("whitegrid")
PALETTE = "viridis"


def load_and_clean(path):
    df = pd.read_csv(path)

    before_rows = len(df)
    df = df.drop_duplicates(subset=["movie_id"]).reset_index(drop=True)
    duplicates_removed = before_rows - len(df)

    # Handle missing values with median imputation for numeric columns
    numeric_cols = ["critic_score", "social_media_buzz", "director_popularity"]
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    print(f"Removed {duplicates_removed} duplicate rows.")
    print(f"Missing values after cleaning:\n{df.isnull().sum()}")

    return df


def plot_genre_distribution(df):
    plt.figure(figsize=(9, 5))
    order = df["genre"].value_counts().index
    sns.countplot(data=df, y="genre", order=order, palette=PALETTE)
    plt.title("Movie Count by Genre")
    plt.xlabel("Number of Movies")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "genre_distribution.png"), dpi=150)
    plt.close()


def plot_budget_vs_revenue(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x="budget_million", y="revenue_million",
        hue="is_success", palette={0: "#e74c3c", 1: "#2ecc71"}, alpha=0.6
    )
    plt.title("Budget vs. Revenue")
    plt.xlabel("Budget (Million USD)")
    plt.ylabel("Revenue (Million USD)")
    plt.legend(title="Success", labels=["Not Successful", "Successful"])
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "budget_vs_revenue.png"), dpi=150)
    plt.close()


def plot_rating_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["audience_rating"], bins=25, kde=True, color="#3498db")
    plt.title("Distribution of Audience Ratings")
    plt.xlabel("Audience Rating (0-10)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "rating_distribution.png"), dpi=150)
    plt.close()


def plot_runtime_vs_success(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="is_success", y="runtime_minutes", palette=PALETTE)
    plt.xticks([0, 1], ["Not Successful", "Successful"])
    plt.title("Runtime vs. Movie Success")
    plt.xlabel("")
    plt.ylabel("Runtime (Minutes)")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "runtime_vs_success.png"), dpi=150)
    plt.close()


def plot_success_rate_by_genre(df):
    plt.figure(figsize=(9, 5))
    rate = df.groupby("genre")["is_success"].mean().sort_values(ascending=False)
    sns.barplot(x=rate.values, y=rate.index, palette=PALETTE)
    plt.title("Success Rate by Genre")
    plt.xlabel("Success Rate")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "success_rate_by_genre.png"), dpi=150)
    plt.close()


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=["number"]).drop(columns=["movie_id"])
    plt.figure(figsize=(11, 8))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()


def plot_success_by_studio_and_director(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    studio_rate = df.groupby("studio_type")["is_success"].mean().sort_values(ascending=False)
    sns.barplot(x=studio_rate.index, y=studio_rate.values, palette=PALETTE, ax=axes[0])
    axes[0].set_title("Success Rate by Studio Type")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_xlabel("")

    tier_order = ["A-List", "Established", "Emerging", "First-Time"]
    tier_rate = df.groupby("director_tier")["is_success"].mean().reindex(tier_order)
    sns.barplot(x=tier_rate.index, y=tier_rate.values, palette=PALETTE, ax=axes[1])
    axes[1].set_title("Success Rate by Director Tier")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_xlabel("")

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "success_by_studio_and_director.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    dataset = load_and_clean(RAW_PATH)
    dataset.to_csv(CLEAN_PATH, index=False)

    plot_genre_distribution(dataset)
    plot_budget_vs_revenue(dataset)
    plot_rating_distribution(dataset)
    plot_runtime_vs_success(dataset)
    plot_success_rate_by_genre(dataset)
    plot_correlation_heatmap(dataset)
    plot_success_by_studio_and_director(dataset)

    print(f"\nClean dataset saved -> {CLEAN_PATH}")
    print(f"Charts saved -> {CHARTS_DIR}")
