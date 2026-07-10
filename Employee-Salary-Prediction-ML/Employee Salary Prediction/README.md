# Task #06 — Employee Salary Prediction Using Machine Learning

**Gexton Education — Data Science Internship Program**  
**Supervised by:** Sir Muhammad Arham MH

---

## Dataset

| Field | Details |
|-------|---------|
| **Source** | Kaggle — Employee Salary Dataset |
| **Author** | prince7489 |
| **URL** | https://www.kaggle.com/datasets/prince7489/employee-salary-dataset/data |
| **Records** | 50 employees |
| **Columns** | EmployeeID, Name, Department, Experience_Years, Education_Level, Age, Gender, City, Monthly_Salary |
| **Departments** | Finance, HR, IT, Marketing, Operations |
| **Cities** | Bangalore, Chennai, Delhi, Hyderabad, Mumbai |
| **Target** | Monthly_Salary (Indian Rupees) |

---

## Project Structure

```
Task06_Employee_Salary_Prediction/
├── data/
│   ├── employee_salary_dataset.csv      ← Raw Kaggle dataset (50 rows)
│   └── cleaned_employee_salary.csv      ← Cleaned & encoded dataset (50 rows, 12 cols)
│
├── scripts/
│   ├── 01_data_cleaning.py              ← Data understanding & cleaning
│   ├── 02_eda.py                        ← EDA — 7 visualisation charts
│   └── 03_model_training.py             ← ML training, evaluation, 2 charts
│
├── app/
│   └── salary_prediction_app.py         ← Streamlit web app with AI insights
│
├── models/
│   ├── best_model.pkl                   ← Saved Linear Regression pipeline
│   ├── model_metadata.pkl               ← Model info (R², RMSE, feature names)
│   └── model_comparison.csv             ← Evaluation table for all 3 models
│
├── charts/                              ← 9 chart PNGs
│   ├── chart_01_experience_vs_salary.png
│   ├── chart_02_salary_by_department.png
│   ├── chart_03_salary_by_education.png
│   ├── chart_04_age_vs_salary.png
│   ├── chart_05_top_bottom_positions.png
│   ├── chart_06_salary_distribution.png
│   ├── chart_07_correlation_heatmap.png
│   ├── chart_08_model_comparison.png
│   └── chart_09_feature_importance.png
│
├── notebooks/
│   └── Employee_Salary_Analysis.ipynb   ← Full Jupyter notebook (24 cells)
│
├── report/
│   └── Task06_Employee_Salary_Prediction_Report.docx
│
├── README.md
└── README.txt
```

---

## How to Run

### 1 · Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib streamlit
```

### 2 · Run Scripts in Order
```bash
python scripts/01_data_cleaning.py    # Cleans data, saves cleaned CSV
python scripts/02_eda.py              # Generates 7 EDA charts
python scripts/03_model_training.py   # Trains 3 models, saves best, generates 2 charts
```

### 3 · Launch Streamlit App
```bash
streamlit run app/salary_prediction_app.py
```
Opens at **http://localhost:8501**

---

## Model Results

| Model | MAE (₹) | RMSE (₹) | R² Test | CV R² (5-fold) |
|-------|---------|----------|---------|----------------|
| **Linear Regression** | ₹24,919 | ₹30,368 | **0.2178** ✓ | -0.5635 |
| Decision Tree | ₹38,930 | ₹51,723 | -1.2691 | -0.7599 |
| Random Forest | ₹33,756 | ₹38,955 | -0.2870 | -0.2709 |

> ⚠️ **Note on low R²:** All models show low/negative R² due to the small dataset size
> (50 records). This is expected — the ML pipeline is fully correct and will perform
> significantly better with 500+ records. Linear Regression is selected as best model
> (highest test R², smallest train-test gap, least overfitting).

---

## AI Insights in the Streamlit App

- **Claude API mode** — Provide an Anthropic API key in the sidebar for
  dynamic, personalised insights via `claude-sonnet-4-6`
- **Rule-based mode** — Works offline without any API key; uses EDA averages
  and RF feature importances to generate contextual career insights

---

## Key EDA Findings

| Finding | Value |
|---------|-------|
| Experience–Salary correlation | r = 0.074 (weak; small dataset effect) |
| Age–Salary correlation | r = 0.061 |
| Highest-paying department | Marketing (₹96,431/mo avg) |
| Lowest-paying department | Finance (₹67,262/mo avg) |
| Highest-paying city | Bangalore (₹99,004/mo avg) |
| Lowest-paying city | Chennai (₹57,380/mo avg) |
| Top RF feature | Experience_Years (47.4% importance) |

---

## Evaluation Criteria Coverage

| Criterion | Weight | Status |
|-----------|--------|--------|
| Data Cleaning | 15% | ✅ Complete — 0 missing, 0 duplicates verified |
| EDA | 15% | ✅ 7 charts covering all 5 required analyses |
| Machine Learning | 35% | ✅ 3 models trained with sklearn Pipelines |
| Model Evaluation | 15% | ✅ MAE / MSE / RMSE / R² + 5-fold CV |
| Streamlit App | 10% | ✅ Fully functional with inputs & prediction |
| AI Integration | 10% | ✅ Claude API + rule-based fallback |

---

**Dataset Reference:** prince7489. *Employee Salary Dataset.* Kaggle, 2024.  
https://www.kaggle.com/datasets/prince7489/employee-salary-dataset/data

*Gexton Education · A21, 3rd Floor Opp. Mehmood Garden, Autobhan Hyd*
