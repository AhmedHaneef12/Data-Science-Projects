"""
Task 05 - Movie Rating Analysis
Gexton Education Summer Internship Program
Supervisor: Sir Muhammad Arham MH

Pipeline: Data Cleaning → Basic Analysis → Visualization
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ── Paths (portable, derived from this file) ──────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE / "data"
CHARTS_DIR = BASE / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

RAW_CSV     = DATA_DIR / "movies_raw.csv"
CLEANED_CSV = DATA_DIR / "movies_cleaned.csv"

# ── Plotting style ─────────────────────────────────────────────────────────────
PALETTE   = ["#2E5C8A", "#E07B39", "#4CAF50", "#9C27B0",
             "#F44336", "#FF9800", "#00BCD4"]
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 – DATA CLEANING
# ════════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1 – DATA CLEANING")
print("=" * 60)

df_raw = pd.read_csv(RAW_CSV)
print(f"\nRaw dataset shape : {df_raw.shape}")
print(f"Missing values:\n{df_raw.isnull().sum()}")
print(f"Duplicate rows    : {df_raw.duplicated().sum()}")

df = df_raw.copy()

# 1a. Remove duplicates
df.drop_duplicates(inplace=True)
print(f"\nAfter removing duplicates: {df.shape}")

# 1b. Drop rows with missing Rating (cannot analyse without it)
df.dropna(subset=["Rating"], inplace=True)
print(f"After dropping missing ratings: {df.shape}")

# 1c. Fill missing Budget with 0 (optional field; 0 flags as unknown)
df["Budget (USD)"] = df["Budget (USD)"].fillna(0)

# 1d. Enforce correct data types
df["Release Year"]      = df["Release Year"].astype(int)
df["Rating"]            = df["Rating"].astype(float)
df["Number of Reviews"] = df["Number of Reviews"].astype(int)
df["Budget (USD)"]      = df["Budget (USD)"].astype(int)

# 1e. Reset index
df.reset_index(drop=True, inplace=True)

print(f"\nCleaned dataset shape: {df.shape}")
print(df.dtypes)

df.to_csv(CLEANED_CSV, index=False)
print(f"\nCleaned CSV saved → {CLEANED_CSV.name}")

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 – BASIC ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2 – BASIC ANALYSIS")
print("=" * 60)

avg_rating = df["Rating"].mean()
print(f"\nAverage rating of all movies : {avg_rating:.2f}")

highest_rated = df.loc[df["Rating"].idxmax(), ["Movie Name", "Rating"]]
print(f"Highest-rated movie          : {highest_rated['Movie Name']} ({highest_rated['Rating']})")

genre_avg = (df.groupby("Genre")["Rating"]
               .mean()
               .round(2)
               .sort_values(ascending=False)
               .reset_index()
               .rename(columns={"Rating": "Avg Rating"}))
print(f"\nGenre-wise average rating:\n{genre_avg.to_string(index=False)}")

top_reviewed = (df.nlargest(5, "Number of Reviews")
                  [["Movie Name", "Number of Reviews"]]
                  .reset_index(drop=True))
print(f"\nTop 5 most-reviewed movies:\n{top_reviewed.to_string(index=False)}")

best_genre   = genre_avg.iloc[0]["Genre"]
best_g_score = genre_avg.iloc[0]["Avg Rating"]
print(f"\nBest performing genre: {best_genre} (avg {best_g_score})")

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 – DATA VISUALISATION
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3 – DATA VISUALISATION")
print("=" * 60)

genres = genre_avg["Genre"].tolist()
colors = PALETTE[:len(genres)]

# ── Chart 1: Genre vs Average Rating (Bar Chart) ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(genre_avg["Genre"], genre_avg["Avg Rating"],
              color=colors, edgecolor="white", linewidth=0.8, width=0.6)

for bar, val in zip(bars, genre_avg["Avg Rating"]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#333333")

ax.axhline(avg_rating, linestyle="--", linewidth=1.5,
           color="#E53935", alpha=0.85, zorder=0)
ax.text(len(genres) - 0.5, avg_rating + 0.07,
        f"Overall Avg: {avg_rating:.2f}",
        fontsize=9, color="#E53935", ha="right")

ax.set_title("Genre vs Average Rating", pad=14)
ax.set_xlabel("Genre")
ax.set_ylabel("Average Rating (out of 10)")
ax.set_ylim(6.5, 10.0)
ax.set_xticklabels(genre_avg["Genre"], rotation=15, ha="right")
plt.tight_layout()
path1 = CHARTS_DIR / "chart1_genre_avg_rating.png"
fig.savefig(path1, bbox_inches="tight")
plt.close()
print(f"Chart 1 saved → {path1.name}")

# ── Chart 2: Top 5 Movies by Rating (Horizontal Bar Chart) ────────────────────
top5 = df.nlargest(5, "Rating")[["Movie Name", "Rating", "Genre"]].reset_index(drop=True)
top5_colors = [PALETTE[genres.index(g) % len(PALETTE)] for g in top5["Genre"]]

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.barh(top5["Movie Name"][::-1], top5["Rating"][::-1],
               color=top5_colors[::-1], edgecolor="white",
               linewidth=0.8, height=0.55)

for bar, val in zip(bars, top5["Rating"][::-1]):
    ax.text(bar.get_width() - 0.12, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", ha="right",
            fontsize=11, fontweight="bold", color="white")

ax.set_title("Top 5 Movies by Rating", pad=14)
ax.set_xlabel("Rating (out of 10)")
ax.set_xlim(7.5, 10.0)
ax.grid(axis="x", alpha=0.3)
ax.grid(axis="y", alpha=0)
plt.tight_layout()
path2 = CHARTS_DIR / "chart2_top5_movies.png"
fig.savefig(path2, bbox_inches="tight")
plt.close()
print(f"Chart 2 saved → {path2.name}")

# ── Chart 3: Rating Distribution (Histogram) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
n, bins, patches = ax.hist(df["Rating"], bins=8, edgecolor="white",
                            linewidth=0.8, color="#2E5C8A", alpha=0.85)

ax.axvline(avg_rating, linestyle="--", linewidth=2,
           color="#E53935", label=f"Mean: {avg_rating:.2f}")
ax.axvline(df["Rating"].median(), linestyle=":",  linewidth=2,
           color="#FF9800", label=f"Median: {df['Rating'].median():.2f}")

ax.set_title("Rating Distribution of Movies", pad=14)
ax.set_xlabel("Rating (out of 10)")
ax.set_ylabel("Number of Movies")
ax.legend(fontsize=10)
plt.tight_layout()
path3 = CHARTS_DIR / "chart3_rating_distribution.png"
fig.savefig(path3, bbox_inches="tight")
plt.close()
print(f"Chart 3 saved → {path3.name}")

# ── Summary stats for report ───────────────────────────────────────────────────
print("\n── SUMMARY FOR REPORT ──────────────────────────────────────────")
print(f"Total movies (cleaned) : {len(df)}")
print(f"Genres covered         : {df['Genre'].nunique()}")
print(f"Release year range     : {df['Release Year'].min()} – {df['Release Year'].max()}")
print(f"Average rating         : {avg_rating:.2f}")
print(f"Rating std dev         : {df['Rating'].std():.2f}")
print(f"Highest-rated movie    : {highest_rated['Movie Name']} ({highest_rated['Rating']})")
print(f"Most-reviewed movie    : {df.loc[df['Number of Reviews'].idxmax(), 'Movie Name']}")
print(f"Best genre             : {best_genre} ({best_g_score})")
high_budget = df[df["Budget (USD)"] > 100_000_000]
print(f"High-budget (>$100M) avg rating : {high_budget['Rating'].mean():.2f}")
low_budget  = df[df["Budget (USD)"].between(1, 100_000_000)]
print(f"Low-budget (<$100M)  avg rating  : {low_budget['Rating'].mean():.2f}")
print("Done.")
