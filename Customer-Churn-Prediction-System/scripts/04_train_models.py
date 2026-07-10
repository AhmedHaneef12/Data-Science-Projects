import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "telco_churn_cleaned.csv"
VIS = BASE_DIR / "charts"
MODELS = BASE_DIR / "models"
REPORT = BASE_DIR / "report"
VIS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["customerID"])

target = "Churn"
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

categorical_cols = [c for c in df.columns
                     if (pd.api.types.is_string_dtype(df[c]) or df[c].dtype == "object")
                     and c != target]
numeric_cols = [c for c in df.columns if c not in categorical_cols + [target]]
print("Categorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)

encoders = {}
df_enc = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    encoders[col] = le

X = df_enc.drop(columns=[target])
y = df_enc[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42),
}

results = {}
fitted = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}
    fitted[name] = model

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    plt.savefig(VIS / f"cm_{fname}.png", dpi=110)
    plt.close()

    print(f"\n{name}")
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))

plt.figure(figsize=(6, 5))
for name, model in fitted.items():
    proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = results[name]["roc_auc"]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(VIS / "08_roc_curves.png", dpi=110)
plt.close()

results_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
print("\n=== Model Comparison ===")
print(results_df.round(4))
results_df.to_csv(REPORT / "model_comparison.csv")

plt.figure(figsize=(7, 4.5))
results_df[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(kind="bar", figsize=(8, 5))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(VIS / "09_model_comparison.png", dpi=110)
plt.close()

best_name = results_df.index[0]
best_model = fitted[best_name]
print(f"\nBest model: {best_name}")

plt.figure(figsize=(7, 6))
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=True)
else:
    importances = pd.Series(np.abs(best_model.coef_[0]), index=X.columns).sort_values(ascending=True)
importances.tail(15).plot(kind="barh", color="#2E86AB")
plt.title(f"Top Feature Importances - {best_name}")
plt.tight_layout()
plt.savefig(VIS / "10_feature_importance.png", dpi=110)
plt.close()

print("\nTop 10 features:")
print(importances.sort_values(ascending=False).head(10))

with open(MODELS / "best_churn_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open(MODELS / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(MODELS / "encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
with open(MODELS / "feature_columns.json", "w") as f:
    json.dump(list(X.columns), f)
with open(MODELS / "numeric_columns.json", "w") as f:
    json.dump(numeric_cols, f)
with open(MODELS / "model_info.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "metrics": {k: float(v) for k, v in results[best_name].items()},
        "all_results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}
    }, f, indent=2)

print("\nSaved model, scaler, encoders, and metadata to", MODELS)
