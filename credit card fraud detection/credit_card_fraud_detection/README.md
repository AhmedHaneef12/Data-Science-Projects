# Credit Card Fraud Detection

An end-to-end machine learning pipeline for identifying fraudulent credit card
transactions. The project covers data cleaning, exploratory data analysis,
class-imbalance correction with SMOTE, feature engineering, model comparison,
hyperparameter tuning, and fraud pattern / business insight analysis, and
ships with a Streamlit app for interactive exploration and scoring.

## Project Structure

```
credit_card_fraud_detection/
├── data/                   # Raw and cleaned transaction datasets
├── scripts/                # Pipeline scripts (run in numbered order)
├── models/                 # Trained model, scaler, and metrics artifacts
├── charts/                 # EDA and model evaluation visualizations
├── notebooks/               # End-to-end Jupyter notebook walkthrough
├── report/                 # Word report and slide presentation
├── app/                    # Streamlit application
├── requirements.txt
└── README.md
```

## Dataset

The working dataset mirrors the structure commonly used for credit card fraud
studies: a `Time` field, 28 anonymized/PCA-style numeric features (`V1`–`V28`),
a transaction `Amount`, and a binary `Class` label (1 = fraud, 0 = legitimate).
It is heavily imbalanced, with fraud accounting for roughly 1.5% of
transactions — a realistic ratio for this type of problem.

`data/cleaned_transactions.csv` ships with the project as the cleaned
deliverable. The raw and feature-engineered versions are regenerated
automatically the first time you run `01_data_preprocessing.py` and
`03_train_models.py`, so they aren't stored in the repository.

## Pipeline

Run the scripts from the `scripts/` folder in order. Each script is
file-anchored, so it can be run from any working directory:

| Script | Purpose |
|---|---|
| `01_data_preprocessing.py` | Builds the transaction dataset and performs cleaning (duplicates, missing values, invalid amounts) |
| `02_eda_visualization.py` | Generates exploratory data analysis charts |
| `03_train_models.py` | Feature engineering, SMOTE balancing, model training, comparison, and tuning |
| `04_fraud_pattern_analysis.py` | Feature importance and fraud pattern / business insight extraction |

```bash
pip install -r requirements.txt

python scripts/01_data_preprocessing.py
python scripts/02_eda_visualization.py
python scripts/03_train_models.py
python scripts/04_fraud_pattern_analysis.py
```

## Models

Three classifiers are trained on a SMOTE-balanced training set and evaluated
on a held-out test set that keeps the original class imbalance:

- Logistic Regression
- Random Forest
- Gradient Boosting

The best-performing baseline is further tuned with `GridSearchCV`. On this
dataset, the tuned **Random Forest** model was selected as the final model,
achieving:

| Metric | Value |
|---|---|
| Precision | 0.67 |
| Recall | 0.52 |
| F1-Score | 0.59 |
| ROC-AUC | 0.98 |

## Streamlit App

An interactive multi-tab application for exploring the dataset, reviewing
model performance, and scoring new transactions:

```bash
streamlit run app/streamlit_app.py
```

Tabs:
- **Overview** — dataset summary and headline model metrics
- **Exploratory Data Analysis** — class distribution, amount, and timing patterns
- **Model Performance** — model comparison, confusion matrix, ROC curve, feature importance
- **Predict a Transaction** — score a new transaction for fraud risk

## Notebook

`notebooks/credit_card_fraud_detection.ipynb` walks through the full pipeline
end-to-end in a single notebook, from raw data through model evaluation and
business insights.

## Report & Presentation

- `report/Credit_Card_Fraud_Detection_Report.docx` — model evaluation and
  business insights report with embedded visuals
- `report/Credit_Card_Fraud_Detection_Presentation.pptx` — 10-slide project
  summary presentation

## Business Insights

- Fraudulent transactions skew toward smaller amounts than legitimate ones,
  consistent with "card testing" behavior ahead of larger fraudulent charges.
- Fraud incidence is concentrated around specific hours of the day, making
  time-of-day a useful risk signal.
- A concentrated subset of features drives most of the model's predictive
  power, supporting a lean, fast-scoring model for near real-time screening.
- The tuned model favors catching more fraud (higher recall) at a reasonable
  cost in false positives — an appropriate trade-off given that missed fraud
  is typically far more costly than a manually reviewed false alarm.

## Requirements

See `requirements.txt`. Python 3.9+ recommended.
