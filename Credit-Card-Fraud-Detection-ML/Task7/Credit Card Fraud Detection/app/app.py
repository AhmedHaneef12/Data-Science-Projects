"""
app.py
-------
Bonus / Creativity Feature - Data Science Task #07
Credit Card Fraud Detection Using Machine Learning

A simple Streamlit demo app that lets a user upload a transaction (or
pick one from the test set) and see the fraud risk score from each of
the 4 trained models.

Run locally with:
    streamlit run app.py

Requires the artifacts produced by scripts/02_model_training.py to
exist in ../models/ (scaler + 4 trained model .joblib files).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="centered")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    models = {name: joblib.load(os.path.join(MODELS_DIR, fname))
              for name, fname in MODEL_FILES.items()}
    return scaler, models


@st.cache_data
def load_sample_data():
    return pd.read_csv(os.path.join(DATA_DIR, "creditcard_cleaned.csv"))


st.title("💳 Credit Card Fraud Detector")
st.caption(
    "Data Science Task #07 — Gexton Education Summer Internship Program. "
    "Dataset: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) "
    "(this demo uses the supplied 1,000-row sample, creditcard_small_1000.csv, "
    "since the full Kaggle file was too large to load in this environment)."
)

try:
    scaler, models = load_artifacts()
    df = load_sample_data()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Please run the scripts in order first:\n\n"
        "`python scripts/01_data_preparation_eda.py`\n"
        "`python scripts/02_model_training.py`\n"
        "`python scripts/03_model_evaluation.py`"
    )
    st.stop()

st.subheader("1. Pick a transaction")
mode = st.radio("Input method", ["Choose a row from the sample dataset", "Enter values manually"])

feature_cols = [c for c in df.columns if c != "Class"]

if mode == "Choose a row from the sample dataset":
    idx = st.slider("Row index", 0, len(df) - 1, 0)
    row = df.iloc[idx]
    st.write("Selected transaction:")
    st.dataframe(row.to_frame().T)
    input_values = row[feature_cols].values.reshape(1, -1)
    true_label = row["Class"]
else:
    st.write("Enter Time and Amount; all V1-V28 values default to 0 (average).")
    time_val = st.number_input("Time", value=50000.0)
    amount_val = st.number_input("Amount ($)", value=50.0, min_value=0.0)
    v_values = {f"V{i}": 0.0 for i in range(1, 29)}
    input_values = np.array([[time_val] + list(v_values.values()) + [amount_val]])
    true_label = None

st.subheader("2. Fraud risk scores")

X_scaled = scaler.transform(pd.DataFrame(input_values, columns=feature_cols))

results = []
for name, model in models.items():
    proba = model.predict_proba(X_scaled)[0, 1]
    pred = model.predict(X_scaled)[0]
    results.append({"Model": name, "Fraud Probability": round(float(proba), 4),
                     "Prediction": "🚨 FRAUD" if pred == 1 else "✅ Legit"})

results_df = pd.DataFrame(results).sort_values("Fraud Probability", ascending=False)
st.dataframe(results_df, use_container_width=True, hide_index=True)

if true_label is not None:
    st.caption(f"Ground-truth label for this row: **{'FRAUD' if true_label == 1 else 'Non-Fraud'}**")

st.divider()
st.caption(
    "⚠️ This is a teaching/demo app trained on a tiny 1,000-row sample with only "
    "2 fraud cases. It is NOT suitable for production use. See the project report "
    "for a full discussion of the dataset's limitations and recommendations."
)
