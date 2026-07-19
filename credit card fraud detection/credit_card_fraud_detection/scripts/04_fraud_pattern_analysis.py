"""Fraud Pattern & Feature Importance Analysis."""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHARTS_DIR = os.path.join(PROJECT_ROOT, "charts")

FEATURED_PATH = os.path.join(DATA_DIR, "featured_transactions.csv")
INSIGHTS_PATH = os.path.join(MODELS_DIR, "business_insights.json")


def plot_feature_importance(model, feature_cols):
    if not hasattr(model, "feature_importances_"):
        return None

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top = importances.sort_values(ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index[::-1], top.values[::-1], color="#2E86AB")
    ax.set_title("Top Features Driving Fraud Predictions", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    path = os.path.join(CHARTS_DIR, "09_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    return top


def main():
    print("=" * 60)
    print("STEP 4: FRAUD PATTERN & FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(FEATURED_PATH)
    model = joblib.load(os.path.join(MODELS_DIR, "fraud_detection_model.joblib"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))
    with open(os.path.join(MODELS_DIR, "model_metrics.json")) as f:
        metrics = json.load(f)

    top_features = plot_feature_importance(model, feature_cols)

    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    amount_summary = {
        "fraud_median_amount": round(float(fraud["Amount"].median()), 2),
        "legit_median_amount": round(float(legit["Amount"].median()), 2),
        "fraud_mean_amount": round(float(fraud["Amount"].mean()), 2),
        "legit_mean_amount": round(float(legit["Amount"].mean()), 2),
    }

    fraud_hour_counts = ((fraud["Time"] % (24 * 3600)) // 3600).value_counts()
    peak_fraud_hour = int(fraud_hour_counts.idxmax()) if len(fraud_hour_counts) else None

    insights = {
        "dataset_size": int(len(df)),
        "fraud_count": int(len(fraud)),
        "fraud_rate_pct": round(len(fraud) / len(df) * 100, 3),
        "amount_summary": amount_summary,
        "peak_fraud_hour": peak_fraud_hour,
        "top_features": top_features.to_dict() if top_features is not None else {},
        "model_metrics": metrics["final_metrics"],
        "best_model_name": metrics["best_model_name"],
    }

    with open(INSIGHTS_PATH, "w") as f:
        json.dump(insights, f, indent=2)

    print(f"\nFraud rate: {insights['fraud_rate_pct']}%")
    print(f"Median fraud amount: {amount_summary['fraud_median_amount']} "
          f"vs. legitimate: {amount_summary['legit_median_amount']}")
    if peak_fraud_hour is not None:
        print(f"Peak fraud hour of day: {peak_fraud_hour}:00")
    print(f"\nBusiness insights saved to: {INSIGHTS_PATH}")


if __name__ == "__main__":
    main()
