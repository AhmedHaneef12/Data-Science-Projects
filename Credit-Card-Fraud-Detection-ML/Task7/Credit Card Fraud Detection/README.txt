CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING
Data Science Task #07 - Gexton Education Summer Internship Program
Supervisor: Sir Muhammad Arham MH

OVERVIEW
--------
This project follows the assignment brief in TASK_7.pdf: load and
clean credit card transaction data, perform exploratory data analysis,
handle severe class imbalance, train multiple classification models,
evaluate them, and provide business recommendations.

DATASET
-------
Original dataset (Kaggle): Credit Card Fraud Detection (mlg-ulb)
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The original Kaggle file (~150 MB, 284,807 transactions) was too large
to open/process in this environment. A 1,000-row random sample
(creditcard_small_1000.csv) was supplied instead and used throughout
this project. That sample contains only 2 fraud cases out of 1,000
transactions; this limitation is discussed in detail in the final
report (report/ folder).

FOLDER STRUCTURE
-----------------
app/        - Streamlit demo app (bonus feature)
charts/     - All EDA & evaluation chart images (PNG)
data/       - Raw sample CSV and cleaned dataset
models/     - Trained model files (.joblib), scaler, train/test splits
notebooks/  - Main deliverable Jupyter notebook (.ipynb), pre-run with outputs
report/     - Final Word report + performance comparison table (CSV)
scripts/    - Standalone Python scripts for each pipeline stage
README.md   - This file in Markdown format
README.txt  - This file in plain text format

HOW TO REPRODUCE
-----------------
1. pip install -r requirements.txt
2. python scripts/01_data_preparation_eda.py
3. python scripts/02_model_training.py
4. python scripts/03_model_evaluation.py
5. python scripts/build_notebook.py   (optional, regenerates the notebook)
6. streamlit run app/app.py           (optional bonus demo app)

The notebook in notebooks/ already contains pre-executed outputs and
can be opened/read directly without rerunning anything.

MODELS TRAINED
----------------
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors

Class imbalance handled via random oversampling of the minority
(fraud) class in the training set, plus class_weight="balanced" where
supported. (SMOTE/imbalanced-learn was unavailable offline, so
sklearn's built-in resample() was used instead.)

BEST MODEL
-----------
Logistic Regression - ranked the single fraud case in the test set as
the #1 highest-risk transaction out of 300 (ROC-AUC = 1.00, PR-AUC =
1.00). See the full report for detailed reasoning, challenges, and
business recommendations.

DELIVERABLES
-------------
- Jupyter Notebook (.ipynb)              -> notebooks/
- Cleaned dataset                        -> data/creditcard_cleaned.csv
- Trained ML models                      -> models/*.joblib
- Confusion matrix visualizations        -> charts/
- Performance comparison table           -> report/model_performance_comparison.csv
- Final project report (DOCX)            -> report/
- GitHub repository link                 -> not created (optional)
