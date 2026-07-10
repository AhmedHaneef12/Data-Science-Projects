Task 04 — Smart Healthcare Data Analysis & Disease Risk Prediction

Gexton Education — Summer Internship Program
Supervisor: Sir Muhammad Arham MH
Dataset: Cardiovascular Disease dataset (Kaggle — sulianova), 70,000 patient records

Project Summary

This project analyzes patient health data to identify the strongest drivers of
cardiovascular disease and builds a machine learning model that predicts each
patient's individual disease risk probability, as requested.

Headline results:
- Cleaned 70,000 → 68,593 records (removed duplicates and physically impossible
  blood pressure / height / weight readings)
- Blood pressure (systolic + diastolic) is the single strongest predictor of
  disease, well ahead of age, cholesterol, weight, and lifestyle factors
- Random Forest model: 73.4% accuracy, 0.80 ROC-AUC
- Every patient now has a model-generated risk probability (0–100%), banded
  into Low / Medium / High

File Guide

| File | Description |
|---|---|
| `data/cardio_train_raw.csv` | Original unmodified dataset (70,000 rows) |
| `data/cardio_train_cleaned.csv` | Cleaned dataset (68,593 rows) — Part 1 |
| `data/cardio_train_features.csv` | Cleaned data + engineered features (BMI, risk_score, risk_group) — Part 3 |
| `data/cardio_risk_predictions.csv` | Per-patient model risk probability & predicted risk band — Part 4 |
| `scripts/01_data_cleaning.py` | Loads, explores, and cleans the raw data |
| `scripts/02_eda.py` | Exploratory data analysis + all distribution/relationship charts |
| `scripts/03_feature_engineering.py` | Builds BMI, BP category, risk score, risk group |
| `scripts/04_modeling.py` | Trains Logistic Regression & Random Forest, generates risk probabilities |
| `charts/*.png` | 12 charts covering distributions, EDA, segmentation, and model evaluation |
| `reports/Task04_Healthcare_Risk_Report.docx` | Full written report covering all 4 task parts |
| `reports/*.txt` | Plain-text logs (cleaning steps, EDA findings, feature engineering, model evaluation) |

How to Reproduce

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python scripts/01_data_cleaning.py
python scripts/02_eda.py
python scripts/03_feature_engineering.py
python scripts/04_modeling.py
```

Each script reads the output of the previous one and is independently runnable
once its input file exists.

Key Findings (see full report for details)

1. Blood pressure dominates. Systolic BP alone accounts for ~48% of the
   Random Forest's predictive power.
2. Age compounds risk — disease rate rises from 23% (age 30–40) to 67%
   (age 61–65).
3. Cholesterol & glucose matter, but less than blood pressure.
4. Lifestyle factors (smoking, alcohol, activity) showed weaker signals
   than expected, likely due to self-reported survey noise.
5. The rule-based risk_score (Part 3) cleanly separates patients: 18.8%
   (Low) → 45.1% (Medium) → 73.3% (High) actual disease rate.
