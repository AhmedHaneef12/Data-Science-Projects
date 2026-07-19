"""Credit Card Fraud Detection — Streamlit App.

Run with: streamlit run app/streamlit_app.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHARTS_DIR = os.path.join(PROJECT_ROOT, "charts")

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, "featured_transactions.csv"))


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "fraud_detection_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))
    with open(os.path.join(MODELS_DIR, "model_metrics.json")) as f:
        metrics = json.load(f)
    insights_path = os.path.join(MODELS_DIR, "business_insights.json")
    insights = None
    if os.path.exists(insights_path):
        with open(insights_path) as f:
            insights = json.load(f)
    return model, scaler, feature_cols, metrics, insights


df = load_data()
model, scaler, feature_cols, metrics, insights = load_artifacts()

st.title("💳 Credit Card Fraud Detection")
st.caption("An end-to-end machine learning system for identifying fraudulent transactions.")

tab_overview, tab_eda, tab_performance, tab_predict = st.tabs(
    ["Overview", "Exploratory Data Analysis", "Model Performance", "Predict a Transaction"]
)

with tab_overview:
    st.header("Project Overview")
    st.write(
        "This application demonstrates a fraud detection pipeline built on transaction-level "
        "data: cleaning and preprocessing, exploratory analysis, class-imbalance handling with "
        "SMOTE, feature engineering, model comparison, and business-facing insights."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Fraud Cases", f"{int(df['Class'].sum()):,}")
    col3.metric("Fraud Rate", f"{df['Class'].mean() * 100:.2f}%")
    col4.metric("Best Model", metrics["best_model_name"])

    st.subheader("Final Model Metrics (Held-out Test Set)")
    m = metrics["final_metrics"]
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Precision", f"{m['precision']:.2f}")
    mcol2.metric("Recall", f"{m['recall']:.2f}")
    mcol3.metric("F1-Score", f"{m['f1_score']:.2f}")
    mcol4.metric("ROC-AUC", f"{m['roc_auc']:.3f}")

    st.subheader("Sample of the Cleaned Dataset")
    st.dataframe(df.head(20), use_container_width=True)

with tab_eda:
    st.header("Exploratory Data Analysis")

    chart_files = [
        ("01_class_distribution.png", "Class Distribution"),
        ("02_amount_distribution.png", "Transaction Amount Patterns"),
        ("03_time_patterns.png", "Transaction Timing Patterns"),
        ("04_correlation_heatmap.png", "Feature Correlation Heatmap"),
        ("05_feature_correlation_with_class.png", "Feature Correlation with Fraud Label"),
    ]

    for filename, caption in chart_files:
        path = os.path.join(CHARTS_DIR, filename)
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)

with tab_performance:
    st.header("Model Performance")

    st.subheader("Baseline Model Comparison")
    comp_df = pd.DataFrame(metrics["baseline_comparison"]).T
    st.dataframe(comp_df, use_container_width=True)

    perf_files = [
        ("08_model_comparison.png", "Model Comparison Across Metrics"),
        ("06_confusion_matrix.png", "Confusion Matrix — Final Model"),
        ("07_roc_curve.png", "ROC Curve — Final Model"),
        ("09_feature_importance.png", "Feature Importance"),
    ]
    cols = st.columns(2)
    for i, (filename, caption) in enumerate(perf_files):
        path = os.path.join(CHARTS_DIR, filename)
        if os.path.exists(path):
            cols[i % 2].image(path, caption=caption, use_container_width=True)

    if insights:
        st.subheader("Business Insights")
        st.markdown(
            f"- **Median fraud amount:** {insights['amount_summary']['fraud_median_amount']} "
            f"vs. **legitimate:** {insights['amount_summary']['legit_median_amount']}\n"
            f"- **Peak fraud hour of day:** {insights['peak_fraud_hour']}:00\n"
            f"- **Fraud rate:** {insights['fraud_rate_pct']}%"
        )

with tab_predict:
    st.header("Score a Transaction")
    st.write(
        "Enter transaction details to estimate fraud risk. The anonymized V1-V28 features "
        "mirror the structure of standard PCA-transformed transaction features; adjust the "
        "ones you have reason to set, and leave the rest at their default (0)."
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Transaction Amount", min_value=0.0, value=50.0, step=1.0)
            hour = st.slider("Hour of Day", 0, 23, 12)
            day = st.number_input("Day Index", min_value=0, value=0, step=1)
        with col2:
            st.write("Key anonymized features (leave at 0 if unknown):")
            v_inputs = {}
            key_v_features = [c for c in feature_cols if c.startswith("V")][:6]
            v_cols_ui = st.columns(3)
            for i, feat in enumerate(key_v_features):
                v_inputs[feat] = v_cols_ui[i % 3].number_input(feat, value=0.0, format="%.2f")

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = {col: 0.0 for col in feature_cols}
        row["Amount"] = amount
        row["Amount_log"] = np.log1p(amount)
        row["Hour"] = hour
        row["Day"] = day
        row["Time"] = day * 24 * 3600 + hour * 3600
        for feat, val in v_inputs.items():
            row[feat] = val

        X_input = pd.DataFrame([row])[feature_cols].values
        X_scaled = scaler.transform(X_input)
        proba = model.predict_proba(X_scaled)[0, 1]
        prediction = model.predict(X_scaled)[0]

        st.subheader("Result")
        if prediction == 1:
            st.error(f"⚠️ Flagged as likely FRAUDULENT — estimated risk: {proba * 100:.1f}%")
        else:
            st.success(f"✅ Likely LEGITIMATE — estimated risk: {proba * 100:.1f}%")
        st.progress(min(max(proba, 0.0), 1.0))
