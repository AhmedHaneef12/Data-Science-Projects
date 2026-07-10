"""
Customer Segmentation and Business Insights
Data Science Task #03
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
from sklearn.cluster import KMeans

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# ============================================================
# 1. DATA UNDERSTANDING AND DATA CLEANING
# ============================================================
print("=" * 60)
print("STEP 1: DATA UNDERSTANDING")
print("=" * 60)

df = pd.read_csv("customer_data_raw.csv")

print(f"\nShape: {df.shape}")
print(f"\nFull dataset:\n{df.to_string()}")
print(f"\nColumn data types:\n{df.dtypes}")
print(f"\nSummary statistics (numerical):\n{df.describe()}")
print(f"\nSummary statistics (categorical):\n{df.describe(include='object')}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")

print("\n" + "=" * 60)
print("STEP 1b: DATA CLEANING")
print("=" * 60)

# Remove exact duplicates
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate row(s)")

# Handle missing AnnualIncome_k: impute with median (robust to outliers)
missing_income = df["AnnualIncome_k"].isnull().sum()
median_income = df["AnnualIncome_k"].median()
df["AnnualIncome_k"] = df["AnnualIncome_k"].fillna(median_income)
print(f"Filled {missing_income} missing AnnualIncome value(s) with median: {median_income}")

# Detect outliers using IQR method
def iqr_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return series[(series < lower) | (series > upper)], lower, upper

income_outliers, low_i, up_i = iqr_outliers(df["AnnualIncome_k"])
print(f"\nIncome outliers (IQR method, bounds {low_i:.1f}-{up_i:.1f}):")
print(income_outliers)
print("Decision: kept as a genuine high-income customer (not a data error) but flagged for review in insights.")

age_outliers, low_a, up_a = iqr_outliers(df["Age"])
print(f"\nAge outliers (IQR method, bounds {low_a:.1f}-{up_a:.1f}): "
      f"{'None found' if age_outliers.empty else age_outliers.to_dict()}")

print(f"\nMissing values after cleaning: {df.isnull().sum().sum()}")
print(f"Final shape: {df.shape}")
print(f"\nCleaned dataset:\n{df.to_string()}")

df.to_csv("customer_data_cleaned.csv", index=False)
print("\nCleaned dataset saved as 'customer_data_cleaned.csv'")

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"\nGender distribution:\n{df['Gender'].value_counts()}")

desc_stats = df[["Age", "AnnualIncome_k", "SpendingScore"]].agg(
    ["mean", "median", "std", "var"]
)
mode_vals = df[["Age", "AnnualIncome_k", "SpendingScore"]].mode().iloc[0]
print(f"\nDescriptive statistics:\n{desc_stats}")
print(f"\nMode values:\n{mode_vals}")

corr = df[["Age", "AnnualIncome_k", "SpendingScore"]].corr()
print(f"\nCorrelation matrix:\n{corr}")

# ============================================================
# 3. DATA VISUALIZATION (8 required charts)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: VISUALIZATIONS")
print("=" * 60)

# Chart 1: Bar chart - Gender count
plt.figure()
df["Gender"].value_counts().plot(kind="bar", color=["#4472C4", "#ED7D31"])
plt.title("Customer Count by Gender", fontsize=14, fontweight="bold")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("chart1_gender_count.png", dpi=150)
plt.close()
print("Chart 1 (bar - gender count): Female and Male customers are fairly balanced in this sample.")

# Chart 2: Histogram - Age distribution
plt.figure()
plt.hist(df["Age"], bins=8, color="#70AD47", edgecolor="black")
plt.title("Age Distribution of Customers", fontsize=14, fontweight="bold")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("chart2_age_histogram.png", dpi=150)
plt.close()
print("Chart 2 (histogram - age): Most customers fall between 20-45 years old, with a smaller older tail (50-60).")

# Chart 3: Pie chart - Income bracket proportions
income_brackets = pd.cut(df["AnnualIncome_k"], bins=[0, 40, 70, 200],
                           labels=["Low (<40k)", "Medium (40-70k)", "High (>70k)"])
plt.figure()
income_brackets.value_counts().plot(kind="pie", autopct="%1.1f%%",
                                      colors=sns.color_palette("Set2"), startangle=140)
plt.title("Customer Share by Income Bracket", fontsize=14, fontweight="bold")
plt.ylabel("")
plt.tight_layout()
plt.savefig("chart3_income_bracket_pie.png", dpi=150)
plt.close()
print("Chart 3 (pie - income brackets): Roughly even split across low, medium, and high income groups.")

# Chart 4: Box plot - SpendingScore by Gender (outlier detection + spread)
plt.figure()
sns.boxplot(data=df, x="Gender", y="SpendingScore", hue="Gender", palette="Set3", legend=False)
plt.title("Spending Score Spread by Gender", fontsize=14, fontweight="bold")
plt.xlabel("Gender")
plt.ylabel("Spending Score (1-100)")
plt.tight_layout()
plt.savefig("chart4_spending_boxplot.png", dpi=150)
plt.close()
print("Chart 4 (box plot - spending by gender): Female customers show a slightly higher median spending score with wider spread.")

# Chart 5: Box plot - Income (outlier detection)
plt.figure()
sns.boxplot(y=df["AnnualIncome_k"], color="#BFBFBF")
plt.title("Annual Income Spread and Outliers", fontsize=14, fontweight="bold")
plt.ylabel("Annual Income (k PKR)")
plt.tight_layout()
plt.savefig("chart5_income_boxplot.png", dpi=150)
plt.close()
print("Chart 5 (box plot - income): One clear high-income outlier (CUST118, 150k) sits well above the upper whisker.")

# Chart 6: Scatter plot - Income vs Spending Score
plt.figure()
plt.scatter(df["AnnualIncome_k"], df["SpendingScore"], s=80, color="#2E75B6", edgecolor="black")
plt.title("Annual Income vs Spending Score", fontsize=14, fontweight="bold")
plt.xlabel("Annual Income (k PKR)")
plt.ylabel("Spending Score (1-100)")
plt.tight_layout()
plt.savefig("chart6_income_vs_spending_scatter.png", dpi=150)
plt.close()
print("Chart 6 (scatter - income vs spending): A clear negative relationship - higher earners tend to have lower spending scores in this sample, except a cluster of low-income, high-spending customers.")

# Chart 7: Scatter plot - Age vs Spending Score
plt.figure()
plt.scatter(df["Age"], df["SpendingScore"], s=80, color="#C00000", edgecolor="black")
plt.title("Age vs Spending Score", fontsize=14, fontweight="bold")
plt.xlabel("Age")
plt.ylabel("Spending Score (1-100)")
plt.tight_layout()
plt.savefig("chart7_age_vs_spending_scatter.png", dpi=150)
plt.close()
print("Chart 7 (scatter - age vs spending): Younger customers tend toward higher spending scores; spending score declines with age.")

# Chart 8: Correlation heatmap
plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0)
plt.title("Correlation Heatmap (Age, Income, Spending Score)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("chart8_correlation_heatmap.png", dpi=150)
plt.close()
print("Chart 8 (heatmap): Age and SpendingScore are negatively correlated; Income and SpendingScore are also negatively correlated.")

# ============================================================
# 4. CUSTOMER SEGMENTATION (K-MEANS CLUSTERING)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: CUSTOMER SEGMENTATION (K-MEANS)")
print("=" * 60)

X = df[["AnnualIncome_k", "SpendingScore"]].copy()
scaler = sklearn.preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method to find optimal k
inertias = []
k_range = range(1, 7)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(list(k_range), inertias, marker="o", color="#2E75B6")
plt.title("Elbow Method for Optimal K", fontsize=14, fontweight="bold")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.tight_layout()
plt.savefig("chart9_elbow_method.png", dpi=150)
plt.close()
print("Elbow chart saved. Inertias by k:", dict(zip(k_range, [round(i, 2) for i in inertias])))

# Based on elbow + small sample size, use k=3
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print(f"\nUsing k={optimal_k} clusters (elbow flattens around k=3 for this dataset).")
print(f"\nCluster assignments:\n{df[['CustomerID', 'AnnualIncome_k', 'SpendingScore', 'Cluster']].to_string()}")

cluster_summary = df.groupby("Cluster").agg(
    Count=("CustomerID", "count"),
    Avg_Age=("Age", "mean"),
    Avg_Income=("AnnualIncome_k", "mean"),
    Avg_SpendingScore=("SpendingScore", "mean")
).round(1)
print(f"\nCluster profiles:\n{cluster_summary}")

# Label clusters by business meaning based on income/spending profile
def label_cluster(row):
    if row["Avg_SpendingScore"] >= 60 and row["Avg_Income"] < 100:
        return "High-Spend / Engaged Shoppers"
    elif row["Avg_Income"] >= 100 and row["Avg_SpendingScore"] < 40:
        return "High Income, Low Engagement (VIP at risk)"
    else:
        return "Conservative (Moderate-High Income, Low Spend)"

cluster_summary["Label"] = cluster_summary.apply(label_cluster, axis=1)
print(f"\nCluster labels:\n{cluster_summary[['Label']]}")

# Chart: Clustered scatter plot
plt.figure()
colors = ["#2E75B6", "#70AD47", "#C00000"]
for c in sorted(df["Cluster"].unique()):
    subset = df[df["Cluster"] == c]
    plt.scatter(subset["AnnualIncome_k"], subset["SpendingScore"],
                s=100, color=colors[c % len(colors)], edgecolor="black",
                label=f"Cluster {c}: {cluster_summary.loc[c, 'Label']}")
plt.title("Customer Segments (K-Means Clustering)", fontsize=14, fontweight="bold")
plt.xlabel("Annual Income (k PKR)")
plt.ylabel("Spending Score (1-100)")
plt.legend(fontsize=8, loc="best")
plt.tight_layout()
plt.savefig("chart10_clusters_scatter.png", dpi=150)
plt.close()
print("\nCluster scatter plot saved.")

df.to_csv("customer_data_segmented.csv", index=False)
print("\nSegmented dataset saved as 'customer_data_segmented.csv'")

print("\nAll analysis complete. Charts and data files saved.")
