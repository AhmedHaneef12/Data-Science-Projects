"""
03_train_models.py

Feature Engineering, Handling Class Imbalance, Model Training & Comparison
----------------------------------------------------------------------------
    1. Feature engineering (time-of-day features, scaling)
    2. Train/test split (stratified, test set kept in original distribution)
    3. Balance the training data with SMOTE (synthetic minority oversampling)
    4. Train and compare Logistic Regression, Random Forest and
       Gradient Boosting classifiers
    5. Hyperparameter tuning of the best-performing model with GridSearchCV
    6. Persist the final model, scaler and metrics to disk
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

# ---------------------------------------------------------------------------
# Path setup (file-anchored, portable across machines)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHARTS_DIR = os.path.join(PROJECT_ROOT, "charts")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

CLEANED_PATH = os.path.join(DATA_DIR, "cleaned_transactions.csv")
FEATURED_PATH = os.path.join(DATA_DIR, "featured_transactions.csv")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()
    df["Hour"] = (df["Time"] % (24 * 3600)) // 3600
    df["Day"] = (df["Time"] // (24 * 3600)).astype(int)
    df["Amount_log"] = np.log1p(df["Amount"])
    return df


# ---------------------------------------------------------------------------
# Minimal SMOTE implementation (k-NN based synthetic oversampling)
# so the pipeline has no dependency beyond scikit-learn.
# ---------------------------------------------------------------------------
def smote_oversample(X, y, minority_label=1, k_neighbors=5, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    X_minority = X[y == minority_label]
    n_minority = len(X_minority)
    n_majority = (y != minority_label).sum()
    n_to_generate = n_majority - n_minority

    if n_to_generate <= 0:
        return X, y

    k = min(k_neighbors, n_minority - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_minority)
    _, neighbor_idx = nn.kneighbors(X_minority)

    synthetic_samples = []
    for _ in range(n_to_generate):
        i = rng.integers(0, n_minority)
        neighbor_choice = rng.integers(1, k + 1)  # skip self at index 0
        j = neighbor_idx[i, neighbor_choice]
        gap = rng.random()
        synthetic = X_minority[i] + gap * (X_minority[j] - X_minority[i])
        synthetic_samples.append(synthetic)

    X_synthetic = np.vstack(synthetic_samples)
    y_synthetic = np.full(n_to_generate, minority_label)

    X_resampled = np.vstack([X, X_synthetic])
    y_resampled = np.concatenate([y, y_synthetic])

    # Shuffle
    shuffle_idx = rng.permutation(len(y_resampled))
    return X_resampled[shuffle_idx], y_resampled[shuffle_idx]


# ---------------------------------------------------------------------------
# Model evaluation helper
# ---------------------------------------------------------------------------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    print(f"\n{name} results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    return metrics, y_pred, y_proba


def main():
    print("=" * 60)
    print("STEP 3: FEATURE ENGINEERING, CLASS BALANCING & MODEL TRAINING")
    print("=" * 60)

    df = pd.read_csv(CLEANED_PATH)
    df = engineer_features(df)
    df.to_csv(FEATURED_PATH, index=False)
    print(f"Feature-engineered dataset saved to: {FEATURED_PATH}")

    feature_cols = [c for c in df.columns if c not in ("Class",)]
    X = df[feature_cols].values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Train fraud rate: {y_train.mean() * 100:.3f}%, Test fraud rate: {y_test.mean() * 100:.3f}%")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nApplying SMOTE to balance the training set...")
    X_train_bal, y_train_bal = smote_oversample(X_train_scaled, y_train)
    print(f"Balanced training set size: {X_train_bal.shape[0]} "
          f"(fraud share: {y_train_bal.mean() * 100:.1f}%)")

    # -----------------------------------------------------------------
    # Train and compare baseline models
    # -----------------------------------------------------------------
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=80, max_depth=2, random_state=RANDOM_STATE),
    }

    results = {}
    fitted_models = {}
    for name, model in candidates.items():
        model.fit(X_train_bal, y_train_bal)
        metrics, y_pred, y_proba = evaluate_model(name, model, X_test_scaled, y_test)
        results[name] = metrics
        fitted_models[name] = model

    best_name = max(results, key=lambda n: results[n]["f1_score"])
    print(f"\nBest baseline model by F1-score: {best_name}")

    # -----------------------------------------------------------------
    # Hyperparameter tuning of the best model
    # -----------------------------------------------------------------
    print(f"\nTuning {best_name} with GridSearchCV...")
    if best_name == "Logistic Regression":
        param_grid = {"C": [0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]}
        base_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    elif best_name == "Random Forest":
        param_grid = {"n_estimators": [100, 150], "max_depth": [8, 12]}
        base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    else:
        param_grid = {"n_estimators": [80, 120], "learning_rate": [0.05, 0.1]}
        base_model = GradientBoostingClassifier(max_depth=2, random_state=RANDOM_STATE)

    grid = GridSearchCV(base_model, param_grid, scoring="f1", cv=3, n_jobs=-1)
    grid.fit(X_train_bal, y_train_bal)
    best_model = grid.best_estimator_
    print(f"Best parameters: {grid.best_params_}")

    final_metrics, y_pred, y_proba = evaluate_model(f"{best_name} (Tuned)", best_model, X_test_scaled, y_test)

    # -----------------------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------------------
    joblib.dump(best_model, os.path.join(MODELS_DIR, "fraud_detection_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.joblib"))
    print(f"\nModel and scaler saved to: {MODELS_DIR}")

    all_results = {
        "baseline_comparison": results,
        "best_model_name": best_name,
        "best_params": grid.best_params_,
        "final_metrics": final_metrics,
        "feature_columns": feature_cols,
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "test_fraud_count": int(y_test.sum()),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Metrics saved to: {METRICS_PATH}")

    # -----------------------------------------------------------------
    # Evaluation charts: confusion matrix, ROC curve, model comparison
    # -----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name} (Tuned)")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "06_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#E63946", linewidth=2, label=f"ROC AUC = {final_metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {best_name} (Tuned)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "07_roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Model comparison bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    metric_names = ["precision", "recall", "f1_score", "roc_auc"]
    x = np.arange(len(metric_names))
    width = 0.25
    for i, (name, m) in enumerate(results.items()):
        vals = [m[k] for k in metric_names]
        ax.bar(x + i * width, vals, width, label=name)
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Precision", "Recall", "F1-Score", "ROC-AUC"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison Across Evaluation Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "08_model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nEvaluation charts saved to charts/ directory.")


if __name__ == "__main__":
    main()
