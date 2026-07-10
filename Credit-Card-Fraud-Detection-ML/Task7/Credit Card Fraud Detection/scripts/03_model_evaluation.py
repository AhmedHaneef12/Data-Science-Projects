"""
03_model_evaluation.py
------------------------
Data Science Task #07 - Credit Card Fraud Detection
Step 3: Model Evaluation & Business Insights

- Loads the 4 trained models + held-out test set
- Computes Accuracy, Precision, Recall, F1-score
- Plots & saves confusion matrices for every model
- Builds a performance comparison table (CSV + PNG)
- Identifies the best performing model
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, average_precision_score
)

MODELS_DIR = "models"
CHARTS_DIR = "charts"
REPORT_DIR = "report"
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. Load test data
# ---------------------------------------------------------------
X_test = pd.read_csv(f"{MODELS_DIR}/X_test.csv")
y_test = pd.read_csv(f"{MODELS_DIR}/y_test.csv").squeeze("columns")

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
}

results = []
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.flatten()

for i, (name, fname) in enumerate(model_files.items()):
    model = joblib.load(f"{MODELS_DIR}/{fname}")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Probability-based ranking metrics (more informative than hard 0/1
    # labels when the test set contains only 1-2 fraud examples)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc_auc = np.nan
    try:
        pr_auc = average_precision_score(y_test, y_proba)
    except ValueError:
        pr_auc = np.nan

    # Where does the model rank the true fraud case(s) among all test
    # transactions, sorted by predicted fraud probability (1 = highest risk)?
    rank_df = pd.DataFrame({"y_true": y_test.values, "proba": y_proba})
    rank_df["rank"] = rank_df["proba"].rank(ascending=False, method="min").astype(int)
    fraud_ranks = rank_df.loc[rank_df.y_true == 1, "rank"].tolist()

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4) if not np.isnan(roc_auc) else np.nan,
        "PR-AUC": round(pr_auc, 4) if not np.isnan(pr_auc) else np.nan,
        "Fraud Rank (of 300)": fraud_ranks,
    })

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["Non-Fraud", "Fraud"], zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Non-Fraud", "Fraud"],
                yticklabels=["Non-Fraud", "Fraud"], ax=axes[i])
    axes[i].set_title(name)
    axes[i].set_ylabel("Actual")
    axes[i].set_xlabel("Predicted")

    # Individual confusion matrix file too
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Non-Fraud", "Fraud"],
                yticklabels=["Non-Fraud", "Fraud"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_").replace("-", "_")
    plt.savefig(f"{CHARTS_DIR}/cm_{safe_name}.png")
    plt.close()

fig.suptitle("Confusion Matrices - All Models", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{CHARTS_DIR}/07_confusion_matrices_all.png")
plt.close(fig)

# ---------------------------------------------------------------
# 2. Performance comparison table
# ---------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values(
    ["F1-Score", "PR-AUC", "ROC-AUC"], ascending=False
).reset_index(drop=True)

# CSV with the full detail (incl. fraud rank list)
results_df.to_csv(f"{REPORT_DIR}/model_performance_comparison.csv", index=False)
print("\nPerformance Comparison Table:\n", results_df)

# Plot comparison table as bar chart (hard-label metrics only)
plt.figure(figsize=(8, 5))
melted = results_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score"]].melt(
    id_vars="Model", var_name="Metric", value_name="Score"
)
sns.barplot(data=melted, x="Model", y="Score", hue="Metric")
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.ylim(0, 1.05)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/08_model_comparison_bar.png")
plt.close()

# Ranking-quality metrics chart (ROC-AUC / PR-AUC) — meaningful even
# with only 1-2 fraud cases in the test set
plt.figure(figsize=(7, 4.5))
melted_auc = results_df[["Model", "ROC-AUC", "PR-AUC"]].melt(
    id_vars="Model", var_name="Metric", value_name="Score"
)
sns.barplot(data=melted_auc, x="Model", y="Score", hue="Metric")
plt.title("Model Ranking Quality: ROC-AUC vs PR-AUC")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/09_roc_pr_auc_comparison.png")
plt.close()

best_model = results_df.iloc[0]["Model"]
print(f"\nBest performing model (by F1-Score, then PR-AUC, then ROC-AUC): {best_model}")

with open(f"{REPORT_DIR}/best_model.txt", "w") as f:
    f.write(best_model)

print("\nStep 3 (Model Evaluation) complete. Charts saved in 'charts/', table saved in 'report/'.")
