"""
Customer Churn Prediction Dashboard
------------------------------------
Run with:  streamlit run app.py
Expects the trained artifacts (model, scaler, encoders) in ../models/
and the cleaned dataset in ../data/processed/
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "processed" / "telco_churn_cleaned.csv"

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(MODELS_DIR / "best_churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open(MODELS_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)
    with open(MODELS_DIR / "numeric_columns.json") as f:
        numeric_columns = json.load(f)
    with open(MODELS_DIR / "model_info.json") as f:
        model_info = json.load(f)
    return model, scaler, encoders, feature_columns, numeric_columns, model_info


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model, scaler, encoders, feature_columns, numeric_columns, model_info = load_artifacts()
df = load_data()

st.title("📉 Customer Churn Prediction Dashboard")
st.caption("Gexton Education — Data Science Task #10 · Customer Churn Prediction System")

tab1, tab2, tab3 = st.tabs(["🔮 Predict Churn", "📊 Customer Insights", "🧠 Model Performance"])

# ---------------------------------------------------------------------------
# TAB 1: Prediction
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Enter customer details to predict churn risk")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", encoders["gender"].classes_)
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Has Partner", encoders["Partner"].classes_)
        dependents = st.selectbox("Has Dependents", encoders["Dependents"].classes_)
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        phone_service = st.selectbox("Phone Service", encoders["PhoneService"].classes_)
        multiple_lines = st.selectbox("Multiple Lines", encoders["MultipleLines"].classes_)
        internet_service = st.selectbox("Internet Service", encoders["InternetService"].classes_)
        online_security = st.selectbox("Online Security", encoders["OnlineSecurity"].classes_)
        online_backup = st.selectbox("Online Backup", encoders["OnlineBackup"].classes_)
        device_protection = st.selectbox("Device Protection", encoders["DeviceProtection"].classes_)

    with col3:
        tech_support = st.selectbox("Tech Support", encoders["TechSupport"].classes_)
        streaming_tv = st.selectbox("Streaming TV", encoders["StreamingTV"].classes_)
        streaming_movies = st.selectbox("Streaming Movies", encoders["StreamingMovies"].classes_)
        contract = st.selectbox("Contract", encoders["Contract"].classes_)
        paperless_billing = st.selectbox("Paperless Billing", encoders["PaperlessBilling"].classes_)
        payment_method = st.selectbox("Payment Method", encoders["PaymentMethod"].classes_)

    col4, col5 = st.columns(2)
    with col4:
        monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 150.0, 65.0, step=0.5)
    with col5:
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges * tenure), step=1.0)

    if st.button("🔍 Predict Churn", type="primary", use_container_width=True):
        raw_input = {
            "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        row = {}
        for col in feature_columns:
            val = raw_input[col]
            if col in encoders:
                val = encoders[col].transform([val])[0]
            row[col] = val
        input_df = pd.DataFrame([row])[feature_columns]
        input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        churn_conf = proba[1]

        st.divider()
        if pred == 1:
            st.error(f"⚠️ **High Risk of Churn** — Confidence: {churn_conf:.1%}")
        else:
            st.success(f"✅ **Likely to Stay** — Confidence: {(1 - churn_conf):.1%}")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Churn Probability", f"{churn_conf:.1%}")
            st.metric("Retention Probability", f"{(1 - churn_conf):.1%}")
        with c2:
            fig, ax = plt.subplots(figsize=(5, 1.2))
            ax.barh([0], [churn_conf], color="#E63946", height=0.5)
            ax.barh([0], [1 - churn_conf], left=[churn_conf], color="#2E86AB", height=0.5)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel("Churn probability")
            st.pyplot(fig)

# ---------------------------------------------------------------------------
# TAB 2: Insights
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Customer Base Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churn Rate", f"{(df['Churn'] == 'Yes').mean():.1%}")
    c3.metric("Avg. Tenure", f"{df['tenure'].mean():.1f} mo")
    c4.metric("Avg. Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")

    st.divider()
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Churn by Contract Type**")
        rate = df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean())
        st.bar_chart(rate)
    with colB:
        st.markdown("**Churn Rate by Internet Service**")
        rate2 = df.groupby("InternetService")["Churn"].apply(lambda s: (s == "Yes").mean())
        st.bar_chart(rate2)

    colC, colD = st.columns(2)
    with colC:
        st.markdown("**Tenure Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=20,
                     palette=["#2E86AB", "#E63946"], ax=ax)
        st.pyplot(fig)
    with colD:
        st.markdown("**Monthly Charges by Churn**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(x="Churn", y="MonthlyCharges", data=df, hue="Churn",
                    palette=["#2E86AB", "#E63946"], legend=False, ax=ax)
        st.pyplot(fig)

    with st.expander("View raw cleaned data sample"):
        st.dataframe(df.head(50), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: Model performance
# ---------------------------------------------------------------------------
with tab3:
    st.subheader(f"Best Model: {model_info['best_model']}")
    m = model_info["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['accuracy']:.1%}")
    c2.metric("Precision", f"{m['precision']:.1%}")
    c3.metric("Recall", f"{m['recall']:.1%}")
    c4.metric("F1 Score", f"{m['f1']:.1%}")
    c5.metric("ROC-AUC", f"{m['roc_auc']:.3f}")

    st.divider()
    st.markdown("**All Models Compared**")
    comp_df = pd.DataFrame(model_info["all_results"]).T
    st.dataframe(comp_df.style.format("{:.3f}").highlight_max(axis=0, color="lightgreen"),
                 use_container_width=True)

    st.caption(
        "Metrics were computed on a held-out 20% test split. The best model was "
        "selected by ROC-AUC. See the project notebook for full evaluation details, "
        "confusion matrices, and feature importance analysis."
    )
