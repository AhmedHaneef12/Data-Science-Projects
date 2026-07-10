# E-commerce Sales Analytics Dashboard

**Data Science Task # 09 — Gexton Education Summer Internship Program**

Dataset: [Global E-Commerce Sales & Customer Data](https://www.kaggle.com/datasets/muhammadaammartufail/global-e-commerce-sales-and-customer-data) (Kaggle)

## Project Overview
Analysis of an e-commerce sales dataset to uncover business insights, identify sales trends,
understand customer purchasing behavior, and build a machine learning model to predict future
sales — presented through an interactive Streamlit dashboard.

## Folder Structure
```
├── data/         Raw and cleaned datasets
│   ├── global_ecommerce_sales.csv
│   ├── global_ecommerce_sales_cleaned.csv
│   └── model_comparison_results.csv
├── scripts/      Standalone Python scripts (cleaning, EDA, ML)
│   ├── analysis.py
│   ├── eda.py
│   └── ml_model.py
├── app/          Streamlit dashboard
│   ├── app.py
│   └── requirements.txt
├── models/       Trained ML model
│   └── sales_prediction_model.pkl
├── charts/       Exported EDA and model evaluation charts (.png)
├── report/       Final project report
│   └── Project_Report.docx
├── notebooks/    Full analysis notebook
│   └── ecommerce_sales_analysis.ipynb
├── README.md
└── README.txt
```

## Key Results
- Total Revenue: $484,559 | Total Profit: $158,872 | Profit Margin: 32.8%
- Top Category: Furniture | Top Region: Europe | Top Payment Method: Credit Card (~40%)
- Best ML Model: Random Forest Regressor (R² = 0.968, MAE = $19.59)

## Run the Dashboard
```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```
Note: `app.py` expects `global_ecommerce_sales_cleaned.csv` and `sales_prediction_model.pkl`
in its working directory — copy them from `data/` and `models/` into `app/`, or update the
file paths in `app.py`, before running.

## Deliverables Checklist
- [x] Source Code
- [x] Jupyter Notebook (.ipynb)
- [x] Cleaned Dataset
- [x] Trained ML Model (.pkl)
- [x] Streamlit Dashboard
- [x] Project Report (docx)
- [ ] GitHub Repository (push this folder to GitHub)
