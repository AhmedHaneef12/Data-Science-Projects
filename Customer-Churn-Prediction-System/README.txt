CUSTOMER CHURN PREDICTION SYSTEM
Data Science Task #10

FOLDERS
-------
app         Streamlit dashboard (app.py)
charts      Exported PNG visualizations from the EDA and model evaluation
data        raw/ and processed/ customer churn datasets
models      Trained model, scaler, encoders, and metadata (pickle/json)
notebooks   Jupyter notebook with the full EDA -> modeling pipeline
report      Project_Report.docx and model_comparison.csv
scripts     Standalone scripts used to build the project (data gen, cleaning,
            EDA, model training) - run in order 01 through 04

FILES
-----
README.md          Full project documentation (setup, results, structure)
requirements.txt   Python dependencies

QUICK START
-----------
1. pip install -r requirements.txt
2. python scripts/01_generate_data.py
3. python scripts/02_clean_data.py
4. python scripts/03_eda.py
5. python scripts/04_train_models.py
6. Open notebooks/Customer_Churn_Prediction.ipynb to see the full analysis
7. cd app and run: streamlit run app.py

See README.md for full details.
