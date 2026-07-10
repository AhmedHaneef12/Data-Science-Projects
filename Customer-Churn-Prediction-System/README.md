# 📉 Customer Churn Prediction System

**Data Science Task #10 — Customer Churn Prediction System**

A complete machine learning project that predicts whether a telecom customer
is likely to churn, based on demographics, account information, and service
usage — including EDA, model comparison, feature importance, and an
interactive Streamlit dashboard.

---

## 🎯 Project Overview

Customer churn (customers leaving a company) is one of the most expensive
problems for subscription-based businesses. This project builds a full
pipeline to:

1. Understand and clean raw customer data
2. Explore the key behavioral and account factors linked to churn
3. Train and compare multiple ML models
4. Select the best model and explain its predictions
5. Serve live predictions through a simple, interactive dashboard

## 🗂️ Repository Structure

```
churn_project/
├── data/
│   ├── raw/telco_churn_raw.csv              # Original dataset
│   └── processed/telco_churn_cleaned.csv    # Cleaned dataset used for modeling
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb      # Full EDA + modeling notebook
├── models/
│   ├── best_churn_model.pkl                 # Trained best model
│   ├── scaler.pkl                           # Fitted StandardScaler
│   ├── encoders.pkl                         # Fitted LabelEncoders per column
│   ├── feature_columns.json                 # Column order expected by model
│   ├── numeric_columns.json                 # Which columns get scaled
│   └── model_info.json                      # Metrics for all trained models
├── app/
│   └── app.py                               # Streamlit dashboard
├── charts/                                  # Exported PNG charts
├── report/
│   ├── Project_Report.docx                  # Written project report
│   └── model_comparison.csv                 # Model metrics table
├── scripts/                                 # Standalone pipeline scripts
├── requirements.txt
└── README.md
```

## 📊 Dataset

The dataset follows the structure of a standard telecom customer churn
dataset: customer demographics, account info (contract, tenure, billing),
and subscribed services (phone, internet, streaming, support add-ons). It
contains 3,000+ customer records with an overall churn rate of ~32%,
consistent with typical telecom industry benchmarks.

## 🧹 Data Preprocessing

- Removed duplicate customer records
- Converted `TotalCharges` from string to numeric, imputing missing values
- Imputed missing `MultipleLines` values using the column mode
- Label-encoded all categorical features
- Standard-scaled numeric features (`tenure`, `MonthlyCharges`,
  `TotalCharges`, `SeniorCitizen`)

## 🔍 Key EDA Findings

- **Contract type is the single biggest churn driver** — month-to-month
  customers churn at ~44%, vs. ~9% for two-year contracts.
- Churned customers have a much lower average tenure (~27 months) than
  retained customers (~41 months).
- Fiber optic customers and customers paying by electronic check churn more.
- Customers without Online Security or Tech Support add-ons are more likely
  to leave.

## 🤖 Models Trained & Compared

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | 76.0% | 65.8% | 51.3% | 57.6% | **0.829** |
| Random Forest | 74.5% | 63.2% | 47.6% | 54.3% | 0.803 |
| Decision Tree | 73.7% | 61.5% | 46.1% | 52.7% | 0.778 |

**Logistic Regression** was selected as the best-performing model based on
ROC-AUC, and is also the most interpretable — a good fit for a business
tool that needs to explain *why* a customer is at risk.

**Top predictive features:** Contract type, Monthly Charges, tenure, Phone
Service, Online Security, Paperless Billing.

## 🖥️ Running the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the pipeline (in order)
```bash
python scripts/01_generate_data.py
python scripts/02_clean_data.py
python scripts/03_eda.py
python scripts/04_train_models.py
```

### 3. Explore the notebook
```bash
jupyter notebook notebooks/Customer_Churn_Prediction.ipynb
```

### 4. Launch the dashboard
```bash
cd app
streamlit run app.py
```

The dashboard has three tabs:
- **Predict Churn** — enter a customer's details and get a churn/stay
  prediction with a confidence score
- **Customer Insights** — churn-rate breakdowns and distribution charts
- **Model Performance** — metrics for all trained models

## 🛠️ Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Streamlit
