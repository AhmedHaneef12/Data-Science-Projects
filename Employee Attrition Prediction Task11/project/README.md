# Employee Attrition Prediction & HR Analytics

An end-to-end machine learning project that analyzes employee data, identifies the key
factors driving employee attrition, and predicts whether an employee is likely to leave
a company. The project covers data cleaning, exploratory data analysis, model training
and comparison, hyperparameter tuning, and business recommendations for HR teams.

## Dataset

**IBM HR Analytics Employee Attrition Dataset**
Source: [Kaggle - IBM HR Analytics Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

The dataset contains 1,470 employee records with 35 attributes covering demographics,
compensation, job role, satisfaction ratings, and tenure.

## Project Structure

```
.
├── app/
│   └── app.py                 # Streamlit app: attrition prediction + HR insights dashboard
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned dataset used for modeling
├── notebooks/
│   └── attrition_analysis.ipynb   # Full analysis notebook (EDA + modeling)
├── scripts/
│   ├── preprocess.py          # Data loading, cleaning, and preprocessing
│   ├── eda.py                 # Exploratory data analysis and chart generation
│   └── train_model.py         # Model training, comparison, and tuning
├── charts/                    # Saved visualization charts (PNG)
├── models/                    # Trained model and preprocessing artifacts
├── report/
│   ├── Employee_Attrition_Documentation.docx   # Full written project report
│   └── model_performance.json                  # Model evaluation metrics
├── requirements.txt
└── README.md
```

All scripts and the app resolve their file paths relative to their own location
(not the current working directory), so they run correctly whether launched from
a terminal, VS Code's Run button, or on a different computer — no path edits needed.

## Approach

1. **Data Collection & Exploration** — Loaded the dataset with Pandas, checked data
   types, missing values, and duplicates, and generated summary statistics.
2. **Data Cleaning** — Removed constant, non-informative columns and the employee ID
   column; confirmed no missing values or duplicates remained.
3. **Exploratory Data Analysis** — Visualized attrition against age, department, job
   role, overtime, income, and other attributes; built a correlation heatmap and
   checked for outliers.
4. **Feature Engineering** — Label-encoded all categorical variables and standardized
   numeric features for scale-sensitive models.
5. **Model Training** — Trained and compared three classification models: Logistic
   Regression, Decision Tree, and Random Forest.
6. **Model Tuning** — Used GridSearchCV with 5-fold cross-validation to tune the
   best-performing model and interpreted feature importance.
7. **Business Insights** — Translated model findings into actionable HR
   recommendations for reducing attrition.

## Model Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.874 | 0.692 | 0.383 | 0.493 |
| Decision Tree | 0.782 | 0.319 | 0.319 | 0.319 |
| Random Forest | 0.844 | 0.545 | 0.128 | 0.207 |

**Best model:** Logistic Regression (tuned with `C=10`, `solver=lbfgs`), selected for
having the strongest F1-score on the minority (attrition = Yes) class.

### Top Predictors of Attrition

OverTime, YearsSinceLastPromotion, Department, NumCompaniesWorked, and
YearsWithCurrManager were the strongest predictors identified by the model.

## Key Business Recommendations

- Manage overtime carefully — it is the single strongest attrition indicator.
- Address career stagnation by improving promotion timelines.
- Review compensation for higher-attrition roles such as Sales Representatives and
  Laboratory Technicians.
- Strengthen onboarding and mentorship for early-tenure employees.
- Track job and environment satisfaction scores as an early warning signal.

## How to Run

1. Clone the repository and open it in VS Code.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the pipeline scripts in order. They can be run from anywhere — the terminal,
   VS Code's Run button, etc. — since each script locates the project folders itself:
   ```
   python scripts/preprocess.py
   python scripts/eda.py
   python scripts/train_model.py
   ```
4. Or open `notebooks/attrition_analysis.ipynb` in Jupyter to explore the full analysis
   interactively.
5. Launch the interactive Streamlit app to predict attrition risk for an individual
   employee and browse the HR insights dashboard:
   ```
   streamlit run app/app.py
   ```

## Streamlit App

The app (`app/app.py`) has three tabs:

- **Predict Attrition** — enter an employee's details and get an attrition risk
  prediction with probability.
- **HR Insights** — browse all EDA charts and key workforce metrics.
- **About the Model** — view the best model, tuned hyperparameters, and full
  performance comparison across all three trained models.

## Deliverables

- Cleaned dataset
- Jupyter notebook with full EDA and modeling workflow
- Data visualization charts
- Trained machine learning model
- Performance evaluation report
- Interactive Streamlit app
- Written project documentation (Word)
- HR business insights and recommendations
