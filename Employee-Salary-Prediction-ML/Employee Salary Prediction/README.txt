==============================================================
  TASK #06 — Employee Salary Prediction Using Machine Learning
  Gexton Education — Data Science Internship Program
  Supervised by: Sir Muhammad Arham MH
==============================================================

DATASET
-------
  Source  : Kaggle — Employee Salary Dataset
  Author  : prince7489
  URL     : https://www.kaggle.com/datasets/prince7489/employee-salary-dataset/data
  Records : 50 employees | Target: Monthly_Salary (Indian Rupees)
  Depts   : Finance, HR, IT, Marketing, Operations
  Cities  : Bangalore, Chennai, Delhi, Hyderabad, Mumbai

HOW TO RUN
----------
  1. Install:
     pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib streamlit

  2. Run scripts in order (from project root):
     python scripts/01_data_cleaning.py
     python scripts/02_eda.py
     python scripts/03_model_training.py

  3. Launch web app:
     streamlit run app/salary_prediction_app.py
     Open: http://localhost:8501

MODEL RESULTS
-------------
  Linear Regression  →  R²=0.2178  |  RMSE=₹30,368  ← BEST
  Decision Tree      →  R²=-1.2691 |  RMSE=₹51,723
  Random Forest      →  R²=-0.2870 |  RMSE=₹38,955

  NOTE: Low R² is due to the small dataset (50 records only).
  The pipeline is correct — larger datasets will produce much higher accuracy.

FILES
-----
  data/employee_salary_dataset.csv         → Raw Kaggle dataset
  data/cleaned_employee_salary.csv         → Cleaned dataset (12 cols)
  scripts/01_data_cleaning.py              → Data cleaning pipeline
  scripts/02_eda.py                        → EDA + 7 chart PNGs
  scripts/03_model_training.py             → 3 ML models + evaluation + 2 charts
  app/salary_prediction_app.py             → Streamlit prediction app
  models/best_model.pkl                    → Saved Linear Regression model
  models/model_comparison.csv             → Performance comparison table
  charts/                                 → 9 chart PNGs (EDA + model results)
  notebooks/Employee_Salary_Analysis.ipynb → Jupyter notebook (24 cells)
  report/Task06_Employee_Salary_Prediction_Report.docx → Full Word report

AI INSIGHTS (STREAMLIT APP)
----------------------------
  Option A: Enter Anthropic API key in sidebar → Claude AI insights
  Option B: Leave blank → Intelligent rule-based insights (offline, free)

REFERENCE
---------
  prince7489. Employee Salary Dataset. Kaggle, 2024.
  https://www.kaggle.com/datasets/prince7489/employee-salary-dataset/data

==============================================================
