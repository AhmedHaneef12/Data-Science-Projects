"""
Model Training Script
Project: Employee Attrition Prediction & HR Analytics

Encodes categorical features, splits the data, trains three classification
models (Logistic Regression, Decision Tree, Random Forest), compares their
performance, tunes the best model, and saves the final model along with
a performance report and feature importance chart.
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Anchor all paths to this script's location so the script works the same way
# whether it's run from a terminal, VS Code's "Run" button, or another machine.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_attrition_data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHARTS_DIR = os.path.join(PROJECT_ROOT, "charts")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")

RANDOM_STATE = 42


def load_and_encode(path):
    df = pd.read_csv(path)

    target = df["Attrition"].map({"Yes": 1, "No": 0})
    features = df.drop(columns=["Attrition"])

    categorical_cols = features.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col])
        encoders[col] = le

    return features, target, encoders


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    print(f"\n--- {name} ---")
    print(classification_report(y_test, preds, target_names=["No", "Yes"]))
    return metrics


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    X, y, encoders = load_and_encode(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # 1. Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    log_reg.fit(X_train_scaled, y_train)
    results["Logistic Regression"] = evaluate_model(
        "Logistic Regression", log_reg, X_test_scaled, y_test
    )

    # 2. Decision Tree
    dtree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dtree.fit(X_train, y_train)
    results["Decision Tree"] = evaluate_model("Decision Tree", dtree, X_test, y_test)

    # 3. Random Forest
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    results["Random Forest"] = evaluate_model("Random Forest", rf, X_test, y_test)

    # Pick best model based on F1-score
    best_model_name = max(results, key=lambda k: results[k]["f1_score"])
    print(f"\nBest performing model (by F1-score): {best_model_name}")

    # Hyperparameter tuning on the best model
    if best_model_name == "Random Forest":
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }
        base_model = RandomForestClassifier(random_state=RANDOM_STATE)
        X_tune_train, X_tune_test = X_train, X_test
    elif best_model_name == "Decision Tree":
        param_grid = {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        }
        base_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
        X_tune_train, X_tune_test = X_train, X_test
    else:
        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"],
        }
        base_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        X_tune_train, X_tune_test = X_train_scaled, X_test_scaled

    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(X_tune_train, y_train)
    tuned_model = grid_search.best_estimator_

    print("\nBest hyperparameters:", grid_search.best_params_)
    tuned_metrics = evaluate_model(
        f"{best_model_name} (Tuned)", tuned_model, X_tune_test, y_test
    )
    results[f"{best_model_name} (Tuned)"] = tuned_metrics

    # Feature importance (tree-based models only)
    if hasattr(tuned_model, "feature_importances_"):
        importances = pd.Series(tuned_model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(8, 6))
        importances.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_title(f"Top 15 Feature Importances - {best_model_name} (Tuned)")
        fig.savefig(
            os.path.join(CHARTS_DIR, "11_feature_importance.png"),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close(fig)

    # Save the final tuned model and preprocessing objects
    joblib.dump(tuned_model, os.path.join(MODELS_DIR, "attrition_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "feature_columns.pkl"))

    with open(os.path.join(MODELS_DIR, "best_model_info.json"), "w") as f:
        json.dump(
            {
                "best_model": best_model_name,
                "best_params": grid_search.best_params_,
                "uses_scaled_input": best_model_name == "Logistic Regression",
            },
            f,
            indent=2,
        )

    # Save performance report as JSON for reference
    with open(os.path.join(REPORT_DIR, "model_performance.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFinal model saved to {MODELS_DIR}/attrition_model.pkl")
    print(f"Performance report saved to {REPORT_DIR}/model_performance.json")


if __name__ == "__main__":
    main()
