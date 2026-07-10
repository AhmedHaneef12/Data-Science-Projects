# Credit Card Fraud Detection Using Machine Learning

**Data Science Task #07 — Gexton Education Summer Internship Program**
Supervisor: Sir Muhammad Arham MH

## Overview

This project builds and evaluates machine learning models to detect
fraudulent credit card transactions, following the assignment brief in
`TASK_7.pdf`. It covers data cleaning, exploratory data analysis (EDA),
class-imbalance handling, training of four classification models, and
model evaluation with business recommendations.

## Dataset

- **Original dataset:** [Credit Card Fraud Detection — Kaggle (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  — 284,807 transactions, 492 fraudulent, features `Time`, `Amount`,
  anonymized PCA components `V1`–`V28`, and target `Class`.
- **Dataset actually used in this project:** `data/creditcard_small_1000_raw.csv`,
  a **1,000-row random sample** supplied by the supervisor, because the
  full Kaggle CSV (~150 MB) was too large to open/process in this
  environment. This sample contains only **2 fraud cases**, which is
  discussed at length in the report as a major limitation.

## Folder Structure

```
Task7/
├── app/          Streamlit demo app (bonus feature) — app.py
├── charts/       All EDA & evaluation visualizations (PNG)
├── data/         Raw sample CSV + cleaned dataset
├── models/       Trained model files (.joblib) + scaler + train/test splits
├── notebooks/    Main deliverable: Credit_Card_Fraud_Detection.ipynb
├── report/       Final project report (DOCX) + performance comparison CSV
├── scripts/      Standalone .py versions of each pipeline stage
├── README.md
└── README.txt
```

## How to Reproduce

```bash
pip install -r requirements.txt

python scripts/01_data_preparation_eda.py   # cleaning + EDA charts
python scripts/02_model_training.py         # preprocessing + imbalance handling + training
python scripts/03_model_evaluation.py       # metrics + confusion matrices + comparison table
python scripts/build_notebook.py            # (re)generates the Jupyter notebook with fresh outputs

streamlit run app/app.py                    # optional bonus demo app
```

The Jupyter notebook in `notebooks/` already contains all outputs
(charts, tables, metrics) pre-executed, so it can simply be opened and
read without re-running anything.

## Models Trained

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors

Class imbalance was handled via **random oversampling** of the
minority (fraud) class in the training set (`sklearn.utils.resample`,
since `imbalanced-learn`/SMOTE was unavailable offline), combined with
`class_weight="balanced"` where supported.

## Headline Results (1,000-row sample, 300-row test set with 1 fraud case)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.997 | 0.00 | 0.00 | 0.00 | **1.00** | **1.00** |
| Decision Tree | 0.997 | 0.00 | 0.00 | 0.00 | 0.50 | 0.0033 |
| K-Nearest Neighbors | 0.997 | 0.00 | 0.00 | 0.00 | 0.50 | 0.0033 |
| Random Forest | 0.997 | 0.00 | 0.00 | 0.00 | 0.49 | 0.0033 |

**Best model: Logistic Regression** — at the standard 0.5 probability
threshold all four models missed the single fraud case in the test
set, but Logistic Regression ranked it as the #1 highest-risk
transaction out of 300 by predicted probability, giving it a perfect
ROC-AUC / PR-AUC. Full discussion, challenges, and business
recommendations are in `report/`.

## Deliverables Checklist (per task brief)

- [x] Jupyter Notebook (`notebooks/Credit_Card_Fraud_Detection.ipynb`)
- [x] Cleaned dataset (`data/creditcard_cleaned.csv`)
- [x] Trained ML models (`models/*.joblib`)
- [x] Confusion matrix visualizations (`charts/cm_*.png`, `charts/07_confusion_matrices_all.png`)
- [x] Performance comparison table (`report/model_performance_comparison.csv`)
- [x] Final project report (`report/*.docx`)
- [ ] GitHub repository link (optional — not created)
