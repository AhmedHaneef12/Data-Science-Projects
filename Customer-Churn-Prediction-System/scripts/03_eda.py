import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "telco_churn_cleaned.csv"
VIS = BASE_DIR / "charts"
VIS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
df = pd.read_csv(DATA_PATH)

plt.figure(figsize=(5, 4))
ax = sns.countplot(x="Churn", data=df, hue="Churn", palette=["#2E86AB", "#E63946"], legend=False)
ax.set_title("Customer Churn Distribution")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom")
plt.tight_layout()
plt.savefig(VIS / "01_churn_distribution.png", dpi=110)
plt.close()

plt.figure(figsize=(6, 4))
ax = sns.countplot(x="Contract", hue="Churn", data=df, palette=["#2E86AB", "#E63946"])
ax.set_title("Churn by Contract Type")
plt.tight_layout()
plt.savefig(VIS / "02_churn_by_contract.png", dpi=110)
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=24, palette=["#2E86AB", "#E63946"])
plt.title("Tenure Distribution by Churn")
plt.tight_layout()
plt.savefig(VIS / "03_tenure_by_churn.png", dpi=110)
plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df, hue="Churn", palette=["#2E86AB", "#E63946"], legend=False)
plt.title("Monthly Charges by Churn")
plt.tight_layout()
plt.savefig(VIS / "04_monthlycharges_by_churn.png", dpi=110)
plt.close()

plt.figure(figsize=(6, 4))
ax = sns.countplot(x="InternetService", hue="Churn", data=df, palette=["#2E86AB", "#E63946"])
ax.set_title("Churn by Internet Service Type")
plt.tight_layout()
plt.savefig(VIS / "05_churn_by_internet.png", dpi=110)
plt.close()

num_df = df[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]].copy()
num_df["Churn"] = (df["Churn"] == "Yes").astype(int)
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(VIS / "06_correlation_heatmap.png", dpi=110)
plt.close()

plt.figure(figsize=(7, 4))
rate = df.groupby("PaymentMethod")["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values()
ax = rate.plot(kind="barh", color="#E63946")
ax.set_xlabel("Churn Rate")
ax.set_title("Churn Rate by Payment Method")
plt.tight_layout()
plt.savefig(VIS / "07_churnrate_by_payment.png", dpi=110)
plt.close()

print("Saved 7 visualization PNGs to", VIS)

overall_rate = (df["Churn"] == "Yes").mean()
print(f"Overall churn rate: {overall_rate:.2%}")
mtm_rate = df[df.Contract == "Month-to-month"]["Churn"].apply(lambda x: x == "Yes").mean()
two_yr_rate = df[df.Contract == "Two year"]["Churn"].apply(lambda x: x == "Yes").mean()
print(f"Month-to-month churn rate: {mtm_rate:.2%}")
print(f"Two-year contract churn rate: {two_yr_rate:.2%}")
avg_tenure_churn = df[df.Churn == "Yes"]["tenure"].mean()
avg_tenure_stay = df[df.Churn == "No"]["tenure"].mean()
print(f"Avg tenure (churned): {avg_tenure_churn:.1f} months | (stayed): {avg_tenure_stay:.1f} months")
