"""
Machine Learning Model Development Script
--------------------------------------------
Trains and evaluates multiple classification models to predict whether
a movie will be a commercial success. Compares performance, selects the
best model, analyzes feature importance, and saves the final pipeline
with joblib.

Outputs:
    models/best_model.pkl
    models/model_comparison.csv
    charts/feature_importance.png
    charts/model_comparison.png
    charts/confusion_matrix.png
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---------------------------------------------------------------
# File-anchored paths
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

CLEAN_PATH = os.path.join(DATA_DIR, "movies_dataset_clean.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.csv")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

RANDOM_STATE = 42

TARGET_COL = "is_success"
NUMERIC_FEATURES = [
    "release_year", "budget_million", "marketing_spend_million",
    "runtime_minutes", "cast_popularity", "director_popularity",
    "num_screens", "critic_score", "social_media_buzz"
]
# Note: `audience_rating` is intentionally excluded from the feature set.
# The success label is partly defined using audience_rating, so including
# it as a predictor would leak the target and produce unrealistically
# perfect scores. It is retained in the dataset for reporting/EDA only.
CATEGORICAL_FEATURES = ["genre", "director_tier", "studio_type"]


def build_preprocessor():
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ])
    return preprocessor


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    return metrics, y_pred


def main():
    df = pd.read_csv(CLEAN_PATH)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()

    candidates = {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1, 10],
                "classifier__solver": ["lbfgs", "liblinear"]
            }
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5]
            }
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "classifier__max_depth": [5, 10, 15, None],
                "classifier__min_samples_split": [2, 5, 10]
            }
        }
    }

    results = []
    fitted_pipelines = {}

    for name, cfg in candidates.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", cfg["estimator"])
        ])

        grid = GridSearchCV(
            pipeline, cfg["param_grid"], cv=5, scoring="f1", n_jobs=-1
        )
        grid.fit(X_train, y_train)

        best_pipeline = grid.best_estimator_
        metrics, y_pred = evaluate_model(name, best_pipeline, X_test, y_test)
        metrics["best_params"] = grid.best_params_
        results.append(metrics)
        fitted_pipelines[name] = best_pipeline

        print(f"\n{name} — best params: {grid.best_params_}")
        print(classification_report(y_test, y_pred, target_names=["Not Successful", "Successful"]))

    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    results_df.to_csv(COMPARISON_PATH, index=False)
    print("\nModel comparison:\n", results_df[["model", "accuracy", "precision", "recall", "f1_score"]])

    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = fitted_pipelines[best_model_name]
    print(f"\nBest model selected: {best_model_name}")

    # Save best model
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Best model saved -> {MODEL_PATH}")

    # Save metrics summary
    best_metrics = [r for r in results if r["model"] == best_model_name][0]
    with open(METRICS_PATH, "w") as f:
        json.dump({
            "best_model": best_model_name,
            "accuracy": best_metrics["accuracy"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1_score": best_metrics["f1_score"],
            "best_params": best_metrics["best_params"]
        }, f, indent=2)

    # ---------------- Charts ----------------
    # Model comparison bar chart
    plt.figure(figsize=(9, 5))
    melted = results_df.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1_score"],
                              var_name="metric", value_name="score")
    sns.barplot(data=melted, x="model", y="score", hue="metric")
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xlabel("")
    plt.ylim(0, 1)
    plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    # Confusion matrix for best model
    y_pred_best = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Successful", "Successful"],
                yticklabels=["Not Successful", "Successful"])
    plt.title(f"Confusion Matrix — {best_model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Feature importance (Random Forest / Decision Tree) or coefficients (Logistic Regression)
    plot_feature_importance(best_pipeline, best_model_name)


def plot_feature_importance(pipeline, model_name):
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    all_feature_names = np.array(NUMERIC_FEATURES + list(cat_feature_names))

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        return

    importance_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(9, 7))
    sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis")
    plt.title(f"Top Feature Importance — {model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
