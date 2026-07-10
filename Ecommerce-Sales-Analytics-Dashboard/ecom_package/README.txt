E-COMMERCE SALES ANALYTICS DASHBOARD
Data Science Task # 09 - Gexton Education Summer Internship Program

Dataset (Kaggle):
https://www.kaggle.com/datasets/muhammadaammartufail/global-e-commerce-sales-and-customer-data

FOLDER STRUCTURE
----------------
data/        - raw and cleaned datasets
scripts/     - cleaning, EDA, and ML training scripts (.py)
app/         - Streamlit dashboard (app.py) and requirements.txt
models/      - trained model (sales_prediction_model.pkl)
charts/      - exported chart images (.png)
report/      - final project report (Project_Report.docx)
notebooks/   - full Jupyter notebook (.ipynb)

KEY RESULTS
-----------
Total Revenue: $484,559
Total Profit: $158,872
Profit Margin: 32.8%
Top Category: Furniture
Top Region: Europe
Top Payment Method: Credit Card (~40% of orders)
Best ML Model: Random Forest Regressor (R2 = 0.968, MAE = $19.59)

RUNNING THE DASHBOARD
----------------------
1. cd app
2. pip install -r requirements.txt
3. Copy global_ecommerce_sales_cleaned.csv (from data/) and sales_prediction_model.pkl
   (from models/) into the app/ folder
4. streamlit run app.py

DELIVERABLES
------------
[x] Source Code
[x] Jupyter Notebook
[x] Cleaned Dataset
[x] Trained ML Model (.pkl)
[x] Streamlit Dashboard
[x] Project Report (docx)
[ ] GitHub Repository - push this folder yourself
