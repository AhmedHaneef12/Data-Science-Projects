"""
build_notebook.py
-------------------
Builds notebooks/Credit_Card_Fraud_Detection.ipynb from scratch as raw
.ipynb JSON (nbformat package is unavailable offline), executing each
code cell in a persistent namespace and embedding REAL captured
stdout / matplotlib image outputs, so the notebook opens already
"run" with genuine results.
"""

import json
import io
import base64
import contextlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NB_PATH = "Credit_Card_Fraud_Detection.ipynb"

import os
os.makedirs("notebooks", exist_ok=True)
os.chdir("notebooks")  # so that "../data", "../models" etc. resolve correctly,
                        # matching the notebook's real location when opened

cells = []
namespace = {}
exec_count = [0]


def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True)
    })


def code(source, capture_images=True):
    exec_count[0] += 1
    n = exec_count[0]
    outputs = []
    buf = io.StringIO()
    plt.close("all")
    try:
        with contextlib.redirect_stdout(buf):
            exec(source, namespace)
    except Exception as e:
        buf.write(f"\n[ERROR] {type(e).__name__}: {e}\n")

    text_out = buf.getvalue()
    if text_out:
        outputs.append({
            "output_type": "stream",
            "name": "stdout",
            "text": text_out.splitlines(keepends=True)
        })

    if capture_images:
        fignums = plt.get_fignums()
        for fn in fignums:
            fig = plt.figure(fn)
            imgbuf = io.BytesIO()
            fig.savefig(imgbuf, format="png", bbox_inches="tight", dpi=100)
            imgbuf.seek(0)
            b64 = base64.b64encode(imgbuf.read()).decode("ascii")
            outputs.append({
                "output_type": "display_data",
                "data": {"image/png": b64, "text/plain": ["<Figure>"]},
                "metadata": {}
            })
        plt.close("all")

    cells.append({
        "cell_type": "code",
        "execution_count": n,
        "metadata": {},
        "outputs": outputs,
        "source": source.splitlines(keepends=True)
    })


# =================================================================
# TITLE & INTRO
# =================================================================
md("""# Credit Card Fraud Detection Using Machine Learning

**Data Science Task #07 — Gexton Education Summer Internship Program**
Supervisor: Sir Muhammad Arham MH

## Project Overview
Credit card fraud is one of the most common financial crimes. Financial
institutions rely on machine learning models to flag suspicious
transactions in real time. In this notebook we:

1. Load and explore the transaction data
2. Clean the data and perform exploratory data analysis (EDA)
3. Preprocess the data and handle severe class imbalance
4. Train four classification models (Logistic Regression, Decision
   Tree, Random Forest, K-Nearest Neighbors)
5. Evaluate and compare the models
6. Draw business conclusions and recommendations

## Dataset
**Original dataset (Kaggle):** [Credit Card Fraud Detection — mlg-ulb](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
284,807 European credit card transactions (Sept 2013), 492 of which are
fraudulent. Features `V1`–`V28` are anonymized principal components
(PCA) of the original transaction data; `Time` and `Amount` are not
transformed.

> **Note on the dataset used in this notebook:** the original Kaggle
> CSV (~150 MB) was too large to load in this environment, so a random
> **1,000-row sample** (`creditcard_small_1000.csv`) was provided by
> the supervisor and used for this assignment instead. The full
> methodology below is exactly what would be applied to the complete
> dataset — only the sample size differs. Because of this, the sample
> contains only **2 fraud cases out of 1,000 transactions**, an even
> more extreme imbalance than the original data (0.2% vs ~0.17%), which
> materially affects what can be concluded statistically (discussed in
> Section 7).
""")

# =================================================================
# 1. IMPORTS
# =================================================================
md("## 1. Import Libraries")

code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
RANDOM_STATE = 42
print("Libraries imported successfully.")
""")

# =================================================================
# 2. LOAD DATA
# =================================================================
md("""## 2. Load the Dataset

Loading the 1,000-row sample provided for this assignment (see note above
about the full Kaggle file being too large for this environment).""")

code("""\
df = pd.read_csv("../data/creditcard_small_1000_raw.csv")
print("Dataset shape:", df.shape)
df.head()
""", capture_images=False)

# Manually add a dataframe display output for df.head()
def add_df_output(varname="df", n=5):
    obj = namespace[varname]
    html = obj.head(n)._repr_html_() if hasattr(obj, "_repr_html_") else str(obj.head(n))
    cells[-1]["outputs"].append({
        "output_type": "execute_result",
        "execution_count": exec_count[0],
        "data": {"text/html": html.splitlines(keepends=True),
                 "text/plain": obj.head(n).to_string().splitlines(keepends=True)},
        "metadata": {}
    })

add_df_output("df")

# =================================================================
# 3. EXPLORE STRUCTURE
# =================================================================
md("## 3. Explore Dataset Structure")

code("""\
print("Columns:", list(df.columns))
print()
print("Info:")
df.info()
""")

code("""\
print("Summary statistics (transposed):")
df.describe().T
""", capture_images=False)
add_df_output_full = None
obj = namespace["df"].describe().T
cells[-1]["outputs"].append({
    "output_type": "execute_result",
    "execution_count": exec_count[0],
    "data": {"text/html": obj._repr_html_().splitlines(keepends=True),
             "text/plain": obj.to_string().splitlines(keepends=True)},
    "metadata": {}
})

# =================================================================
# 4. MISSING VALUES & DUPLICATES
# =================================================================
md("## 4. Check Missing Values & Duplicate Records")

code("""\
n_missing = df.isnull().sum().sum()
n_duplicates = df.duplicated().sum()
print(f"Total missing values: {n_missing}")
print(f"Total duplicate rows: {n_duplicates}")
""")

# =================================================================
# 5. CLEANING
# =================================================================
md("""## 5. Clean the Dataset

No missing values or duplicates were found in this sample, but the
cleaning step is included for completeness / reproducibility on the
full dataset.""")

code("""\
df_clean = df.copy()
if n_missing > 0:
    df_clean = df_clean.dropna()
if n_duplicates > 0:
    df_clean = df_clean.drop_duplicates()

df_clean["Class"] = df_clean["Class"].astype(int)
os.makedirs("../data", exist_ok=True)
df_clean.to_csv("../data/creditcard_cleaned.csv", index=False)
print("Cleaned dataset shape:", df_clean.shape)
print("Saved to ../data/creditcard_cleaned.csv")
""")

# =================================================================
# 6. EDA
# =================================================================
md("## 6. Exploratory Data Analysis (EDA)")
md("### 6.1 Class Distribution: Fraud vs Non-Fraud")

code("""\
class_counts = df_clean["Class"].value_counts().sort_index()
fraud_pct = class_counts.get(1, 0) / len(df_clean) * 100
print(class_counts)
print(f"Fraud percentage: {fraud_pct:.4f}%")

plt.figure(figsize=(5, 4))
ax = sns.countplot(x="Class", data=df_clean, hue="Class",
                    palette={0: "#2E75B6", 1: "#D9534F"}, legend=False)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Non-Fraud (0)", "Fraud (1)"])
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom")
plt.title("Class Distribution: Fraud vs Non-Fraud Transactions")
plt.ylabel("Number of Transactions"); plt.xlabel("")
plt.tight_layout()
plt.show()
""")

md("""**Insight:** The dataset is extremely imbalanced — only 2 of 1,000
transactions (0.2%) are fraudulent. This mirrors real-world fraud data,
where fraud is always a tiny minority of all transactions, and is the
central challenge of this project.""")

md("### 6.2 Transaction Amount Distribution")

code("""\
plt.figure(figsize=(6, 4))
sns.histplot(df_clean["Amount"], bins=40, color="#2E75B6", kde=True)
plt.title("Distribution of Transaction Amount")
plt.xlabel("Amount ($)"); plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print(df_clean["Amount"].describe())
""")

md("### 6.3 Relationship Between Transaction Amount and Fraud")

code("""\
plt.figure(figsize=(5, 4))
sns.boxplot(x="Class", y="Amount", data=df_clean, hue="Class",
            palette={0: "#2E75B6", 1: "#D9534F"}, legend=False)
plt.xticks([0, 1], ["Non-Fraud", "Fraud"])
plt.title("Transaction Amount by Class")
plt.tight_layout()
plt.show()

print(df_clean.groupby("Class")["Amount"].describe())
""")

md("""**Insight:** In this sample, the two fraudulent transactions had
amounts of \\$364.19 and \\$520.12 — both above the median non-fraud
amount (\\$20.12) but well within the normal range, so amount alone is
not a reliable standalone fraud signal here (consistent with findings
on the full Kaggle dataset).""")

md("### 6.4 Transaction Time Distribution")

code("""\
plt.figure(figsize=(6, 4))
sns.histplot(df_clean["Time"], bins=40, color="#5B9BD5")
plt.title("Distribution of Transaction Time")
plt.xlabel("Time (seconds since first transaction)"); plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
""")

md("### 6.5 Correlation Heatmap")

code("""\
plt.figure(figsize=(12, 10))
corr = df_clean.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
plt.title("Correlation Heatmap of All Features")
plt.tight_layout()
plt.show()
""")

md("### 6.6 Feature Correlation with Fraud (Class)")

code("""\
class_corr = corr["Class"].drop("Class").sort_values()
plt.figure(figsize=(7, 8))
class_corr.plot(kind="barh", color=np.where(class_corr > 0, "#D9534F", "#2E75B6"))
plt.title("Feature Correlation with Fraud (Class)")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.show()

print("Top 5 positively correlated features with fraud:")
print(class_corr.sort_values(ascending=False).head(5))
print()
print("Top 5 negatively correlated features with fraud:")
print(class_corr.head(5))
""")

md("""### EDA Summary

- The dataset is **severely imbalanced**: only 0.2% of transactions in
  this sample are fraudulent (2 of 1,000), which is even more extreme
  than the ~0.17% fraud rate in the full Kaggle dataset.
- No missing values or duplicate rows were found.
- Transaction **amount** is right-skewed with most transactions under
  \\$100; the two fraud cases fell in the \\$300–\\$520 range, i.e. above
  the typical (median) transaction but not unusually large overall.
- Several PCA features (e.g. V14, V12, V10, V17 on the full dataset)
  are known to correlate strongly with fraud; in this small sample the
  correlations are noisy due to only having 2 fraud examples, but the
  same anonymized features drive the signal.
- **Time** shows a cyclical, bimodal pattern consistent with daily
  transaction volume cycles.
""")

# =================================================================
# 7. PREPROCESSING + IMBALANCE HANDLING
# =================================================================
md("""## 7. Prepare Data for Machine Learning

### 7.1 Feature Scaling
All features (including `Time` and `Amount`, which are on a very
different scale than the PCA components `V1`-`V28`) are standardized
with `StandardScaler` — important for distance-based models like KNN
and for Logistic Regression convergence.""")

code("""\
X = df_clean.drop(columns=["Class"])
y = df_clean["Class"]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

os.makedirs("../models", exist_ok=True)
joblib.dump(scaler, "../models/scaler.joblib")
print("Features scaled. Shape:", X_scaled.shape)
""")

md("""### 7.2 Train / Test Split

We use a **stratified** split so that the (very few) fraud cases are
distributed proportionally between train and test sets.""")

code("""\
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
print("Train class distribution:")
print(y_train.value_counts())
print()
print("Test class distribution:")
print(y_test.value_counts())
""")

md("""### 7.3 Handling Class Imbalance

The task requires handling the imbalanced classes via oversampling,
undersampling, or class weighting. We apply **both**:

1. **Random oversampling** of the minority (fraud) class in the
   *training set only* (using `sklearn.utils.resample`), so models
   that don't natively support class weights (like KNN) still see a
   balanced training signal.
   *(Note: SMOTE from `imbalanced-learn` was not available in this
   offline environment, so sklearn's built-in resampling is used
   instead — one of the techniques explicitly permitted by the task
   brief.)*
2. **`class_weight="balanced"`** for Logistic Regression, Decision
   Tree, and Random Forest as an additional safeguard.

Oversampling is applied **after** the train/test split and **only to
the training data** to avoid data leakage into the test set.""")

code("""\
train_df = X_train.copy()
train_df["Class"] = y_train.values

majority = train_df[train_df.Class == 0]
minority = train_df[train_df.Class == 1]

minority_upsampled = resample(
    minority, replace=True, n_samples=len(majority), random_state=RANDOM_STATE
)

train_balanced = pd.concat([majority, minority_upsampled]).sample(
    frac=1, random_state=RANDOM_STATE
).reset_index(drop=True)

X_train_bal = train_balanced.drop(columns=["Class"])
y_train_bal = train_balanced["Class"]

print("Balanced training class distribution:")
print(y_train_bal.value_counts())
""")

# =================================================================
# 8. MODEL TRAINING
# =================================================================
md("""## 8. Train Classification Models

Four classification models are trained on the oversampled training
data: **Logistic Regression**, **Decision Tree**, **Random Forest**,
and **K-Nearest Neighbors**.""")

code("""\
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, max_depth=6
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, max_depth=8
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
}

for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    fname = name.lower().replace(" ", "_").replace("-", "_")
    joblib.dump(model, f"../models/{fname}.joblib")
    print(f"Trained: {name}")

print("\\nAll 4 models trained and saved to ../models/")
""")

# =================================================================
# 9. EVALUATION
# =================================================================
md("""## 9. Model Evaluation

Each model is evaluated on the **untouched, original-distribution**
test set using Accuracy, Precision, Recall, F1-Score and a Confusion
Matrix, as required. We also report **ROC-AUC** and **PR-AUC** computed
from predicted probabilities, since with only 1 fraud case in the test
set, hard 0/1 classification metrics alone are not very informative
(see discussion below).""")

code("""\
results = []
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.flatten()

for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    y_proba = model.predict_proba(X_test)[:, 1]
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc_auc = np.nan
    try:
        pr_auc = average_precision_score(y_test, y_proba)
    except ValueError:
        pr_auc = np.nan

    rank_df = pd.DataFrame({"y_true": y_test.values, "proba": y_proba})
    rank_df["rank"] = rank_df["proba"].rank(ascending=False, method="min").astype(int)
    fraud_ranks = rank_df.loc[rank_df.y_true == 1, "rank"].tolist()

    results.append({"Model": name, "Accuracy": round(acc,4), "Precision": round(prec,4),
                     "Recall": round(rec,4), "F1-Score": round(f1,4),
                     "ROC-AUC": round(roc_auc,4), "PR-AUC": round(pr_auc,4),
                     "Fraud Rank (of 300)": fraud_ranks})

    print(f"=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["Non-Fraud","Fraud"], zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Non-Fraud","Fraud"], yticklabels=["Non-Fraud","Fraud"], ax=axes[i])
    axes[i].set_title(name); axes[i].set_ylabel("Actual"); axes[i].set_xlabel("Predicted")

fig.suptitle("Confusion Matrices - All Models", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
""")

md("### 9.1 Performance Comparison Table")

code("""\
results_df = pd.DataFrame(results).sort_values(
    ["F1-Score", "PR-AUC", "ROC-AUC"], ascending=False
).reset_index(drop=True)

os.makedirs("../report", exist_ok=True)
results_df.to_csv("../report/model_performance_comparison.csv", index=False)
results_df
""", capture_images=False)
obj = namespace["results_df"]
cells[-1]["outputs"].append({
    "output_type": "execute_result",
    "execution_count": exec_count[0],
    "data": {"text/html": obj._repr_html_().splitlines(keepends=True),
             "text/plain": obj.to_string().splitlines(keepends=True)},
    "metadata": {}
})

md("### 9.2 Visual Comparison")

code("""\
plt.figure(figsize=(8, 5))
melted = results_df[["Model","Accuracy","Precision","Recall","F1-Score"]].melt(
    id_vars="Model", var_name="Metric", value_name="Score"
)
sns.barplot(data=melted, x="Model", y="Score", hue="Metric")
plt.title("Model Performance Comparison (Hard-Label Metrics)")
plt.ylabel("Score"); plt.xticks(rotation=15); plt.ylim(0, 1.05)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
""")

code("""\
plt.figure(figsize=(7, 4.5))
melted_auc = results_df[["Model","ROC-AUC","PR-AUC"]].melt(
    id_vars="Model", var_name="Metric", value_name="Score"
)
sns.barplot(data=melted_auc, x="Model", y="Score", hue="Metric")
plt.title("Model Ranking Quality: ROC-AUC vs PR-AUC")
plt.ylabel("Score"); plt.xticks(rotation=15); plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()
""")

best_model = namespace["results_df"].iloc[0]["Model"]

md(f"""## 10. Best-Performing Model

Based on F1-Score (with PR-AUC / ROC-AUC as tiebreakers), the
best-performing model in this evaluation is **{best_model}**.

**Why it performed better:**
- Logistic Regression assigned the single fraudulent test transaction
  the **highest predicted fraud probability of all 300 test
  transactions** (rank #1), giving it a perfect ROC-AUC and PR-AUC of
  1.0 on this test set — meaning that if a bank used this model's
  *risk score* to flag the top few highest-risk transactions for
  manual review (rather than a hard 0.5 cutoff), it would have caught
  this fraud case.
- The tree-based models (Decision Tree, Random Forest) and KNN, after
  being trained on a *duplicated* (oversampled) single fraud example,
  tend to overfit narrowly to that one example's exact feature values,
  which hurts their ability to generalize/rank a *different* fraud
  example correctly in the test set.
- At the standard 0.5 probability threshold, **all four models still
  missed the single fraud case in the test set** (Recall = 0,
  Precision = 0, F1 = 0) — accuracy alone (99.67%) is a highly
  misleading metric here, since simply predicting "non-fraud" for
  every transaction would also score 99.67% accuracy. This is exactly
  why **Precision, Recall, F1-Score, and ranking metrics (ROC-AUC /
  PR-AUC) — not Accuracy — must be used to judge fraud models.**
""")

# =================================================================
# 11. CHALLENGES & BUSINESS RECOMMENDATIONS
# =================================================================
md("""## 11. Challenges of Detecting Fraudulent Transactions

1. **Extreme class imbalance** — fraud is always a tiny fraction of
   all transactions (0.2% in this sample; ~0.17% in the full Kaggle
   dataset). Models can achieve very high accuracy by simply never
   predicting fraud, which is useless in practice.
2. **Very few positive examples to learn from** — with only 1-2 fraud
   cases available for training/testing in this reduced sample, models
   cannot reliably learn the general patterns of fraud; results here
   should be treated as a **methodology demonstration**, not a
   statistically robust fraud model. On the full 284,807-row dataset
   (492 fraud cases) these same techniques would produce far more
   reliable and generalizable results.
3. **Anonymized/PCA features (V1-V28)** make the model accurate but
   hard to interpret or explain to compliance/regulatory teams, since
   the original meaning of each feature is hidden.
4. **Evolving fraud patterns** — fraudsters constantly change their
   tactics, so a model trained on historical data can go stale quickly
   and needs continuous retraining/monitoring.
5. **Cost asymmetry** — a false negative (missed fraud) is usually far
   more costly than a false positive (a legitimate transaction flagged
   for review), so models should be tuned/thresholded with this
   asymmetry in mind rather than optimizing for plain accuracy.

## 12. Business Recommendations for Financial Institutions

1. **Never use Accuracy alone** to judge a fraud model — track
   Precision, Recall, F1-Score, and PR-AUC, and choose a decision
   threshold based on the institution's tolerance for false
   positives vs. false negatives.
2. **Use risk scores, not just binary flags** — rank transactions by
   predicted fraud probability and route the highest-risk transactions
   to manual review or step-up authentication, rather than relying on
   a single hard cutoff.
3. **Retrain models regularly** on recent transaction data to keep up
   with evolving fraud patterns, and monitor for model/data drift.
4. **Combine multiple models/signals** (ensemble approaches, rule-based
   checks, device/location signals) since no single model will catch
   every type of fraud.
5. **Invest in collecting more labeled fraud examples** — the single
   biggest lever for improving real-world fraud models is more
   (recent) positive examples, since fraud detection is fundamentally
   data-starved by nature.
6. **Use class-imbalance techniques in production** (oversampling,
   undersampling, class weighting, or anomaly-detection approaches)
   exactly as done in this notebook, scaled to the full dataset.

## 13. Conclusion

This notebook implemented a complete, end-to-end fraud detection
pipeline: data loading, cleaning, EDA, feature scaling, stratified
train/test splitting, class-imbalance handling via oversampling and
class weighting, training of four classification models, and
evaluation using Accuracy, Precision, Recall, F1-Score, Confusion
Matrices, and ranking metrics (ROC-AUC / PR-AUC).

**Logistic Regression** was selected as the best-performing model in
this evaluation, successfully ranking the test set's fraud case as the
single highest-risk transaction out of 300. The severe scarcity of
fraud examples in this 1,000-row sample (only 2 total) is itself the
most important finding of this analysis: it demonstrates, very
concretely, why fraud detection is such a hard real-world machine
learning problem, and why financial institutions need large volumes of
historical fraud data, careful metric selection, and risk-based
(rather than purely binary) decisioning to deploy these models
successfully.

---
**Dataset source:** [Kaggle — Credit Card Fraud Detection (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
**Sample used:** `creditcard_small_1000.csv` (1,000-row random sample, supplied because the full ~150MB file could not be opened in this environment)
""")

# =================================================================
# WRITE THE NOTEBOOK
# =================================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(NB_PATH, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {NB_PATH} with {len(cells)} cells.")
