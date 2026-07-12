"""
Streamlit App
Project: Employee Attrition Prediction & HR Analytics

Interactive app for predicting employee attrition risk using the trained
model, and for browsing the EDA charts and dataset insights.
"""

import os
import json

import joblib
import pandas as pd
import streamlit as st

# Anchor all paths to this script's location so the app works the same way
# regardless of the machine or the folder it is launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHARTS_DIR = os.path.join(PROJECT_ROOT, "charts")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_attrition_data.csv")

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "attrition_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    with open(os.path.join(MODELS_DIR, "best_model_info.json")) as f:
        model_info = json.load(f)
    return model, scaler, encoders, feature_columns, model_info


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model, scaler, encoders, feature_columns, model_info = load_artifacts()
df = load_data()

st.title("Employee Attrition Prediction & HR Analytics")
st.caption(
    "Dataset: IBM HR Analytics Employee Attrition Dataset — "
    "[source](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)"
)

tab_predict, tab_insights, tab_about = st.tabs(
    ["🔮 Predict Attrition", "📊 HR Insights", "ℹ️ About the Model"]
)

# --------------------------------------------------------------------------
# TAB 1: Prediction
# --------------------------------------------------------------------------
with tab_predict:
    st.subheader("Predict attrition risk for an employee")
    st.write("Enter employee details below to estimate the likelihood of attrition.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 60, 35)
        department = st.selectbox("Department", encoders["Department"].classes_)
        job_role = st.selectbox("Job Role", encoders["JobRole"].classes_)
        business_travel = st.selectbox("Business Travel", encoders["BusinessTravel"].classes_)
        marital_status = st.selectbox("Marital Status", encoders["MaritalStatus"].classes_)
        gender = st.selectbox("Gender", encoders["Gender"].classes_)
        education_field = st.selectbox("Education Field", encoders["EducationField"].classes_)
        education = st.slider("Education Level (1=Below College, 5=Doctorate)", 1, 5, 3)

    with col2:
        monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=100)
        daily_rate = st.number_input("Daily Rate", 100, 1500, 800, step=10)
        hourly_rate = st.number_input("Hourly Rate", 30, 100, 65)
        monthly_rate = st.number_input("Monthly Rate", 2000, 27000, 14000, step=100)
        percent_salary_hike = st.slider("Percent Salary Hike", 11, 25, 15)
        stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        overtime = st.selectbox("OverTime", encoders["OverTime"].classes_)
        distance_from_home = st.slider("Distance From Home (miles)", 1, 30, 10)

    with col3:
        total_working_years = st.slider("Total Working Years", 0, 40, 10)
        years_at_company = st.slider("Years At Company", 0, 40, 5)
        years_in_current_role = st.slider("Years In Current Role", 0, 20, 3)
        years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        years_with_curr_manager = st.slider("Years With Current Manager", 0, 20, 3)
        num_companies_worked = st.slider("Number of Companies Worked", 0, 10, 2)
        training_times_last_year = st.slider("Training Times Last Year", 0, 6, 2)
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])

    st.markdown("**Satisfaction & Involvement Ratings** (1 = Low, 4 = High)")
    col4, col5, col6, col7, col8 = st.columns(5)
    with col4:
        env_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2)
    with col5:
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2)
    with col6:
        relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], index=2)
    with col7:
        job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=2)
    with col8:
        work_life_balance = st.selectbox("Work-Life Balance", [1, 2, 3, 4], index=2)

    performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4], index=2)

    if st.button("Predict Attrition Risk", type="primary"):
        input_dict = {
            "Age": age,
            "BusinessTravel": encoders["BusinessTravel"].transform([business_travel])[0],
            "DailyRate": daily_rate,
            "Department": encoders["Department"].transform([department])[0],
            "DistanceFromHome": distance_from_home,
            "Education": education,
            "EducationField": encoders["EducationField"].transform([education_field])[0],
            "EnvironmentSatisfaction": env_satisfaction,
            "Gender": encoders["Gender"].transform([gender])[0],
            "HourlyRate": hourly_rate,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level,
            "JobRole": encoders["JobRole"].transform([job_role])[0],
            "JobSatisfaction": job_satisfaction,
            "MaritalStatus": encoders["MaritalStatus"].transform([marital_status])[0],
            "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies_worked,
            "OverTime": encoders["OverTime"].transform([overtime])[0],
            "PercentSalaryHike": percent_salary_hike,
            "PerformanceRating": performance_rating,
            "RelationshipSatisfaction": relationship_satisfaction,
            "StockOptionLevel": stock_option_level,
            "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times_last_year,
            "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_current_role,
            "YearsSinceLastPromotion": years_since_last_promotion,
            "YearsWithCurrManager": years_with_curr_manager,
        }

        input_df = pd.DataFrame([input_dict])[feature_columns]

        if model_info.get("uses_scaled_input"):
            model_input = scaler.transform(input_df)
        else:
            model_input = input_df

        prediction = model.predict(model_input)[0]
        probability = model.predict_proba(model_input)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ High Attrition Risk — estimated probability: {probability:.1%}")
        else:
            st.success(f"✅ Low Attrition Risk — estimated probability: {probability:.1%}")
        st.progress(min(float(probability), 1.0))

# --------------------------------------------------------------------------
# TAB 2: HR Insights (EDA charts)
# --------------------------------------------------------------------------
with tab_insights:
    st.subheader("Exploratory Data Analysis & Key Drivers")

    attrition_rate = (df["Attrition"] == "Yes").mean() * 100
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Employees", len(df))
    m2.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    m3.metric("Avg. Monthly Income", f"${df['MonthlyIncome'].mean():,.0f}")

    chart_files = [
        ("01_attrition_distribution.png", "Overall Attrition Distribution"),
        ("02_age_distribution.png", "Age Distribution by Attrition"),
        ("03_attrition_by_department.png", "Attrition by Department"),
        ("04_attrition_by_jobrole.png", "Attrition by Job Role"),
        ("05_attrition_by_overtime.png", "Attrition by OverTime Status"),
        ("06_income_vs_attrition_boxplot.png", "Monthly Income vs Attrition"),
        ("07_correlation_heatmap.png", "Correlation Heatmap"),
        ("08_attrition_by_gender.png", "Attrition by Gender"),
        ("09_attrition_by_worklifebalance.png", "Attrition by Work-Life Balance"),
        ("10_distance_from_home.png", "Distance From Home Distribution"),
        ("11_feature_importance.png", "Top Feature Importances"),
    ]

    cols = st.columns(2)
    for i, (filename, caption) in enumerate(chart_files):
        path = os.path.join(CHARTS_DIR, filename)
        if os.path.exists(path):
            with cols[i % 2]:
                st.image(path, caption=caption, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 3: About the model
# --------------------------------------------------------------------------
with tab_about:
    st.subheader("Model Details")
    st.write(f"**Best model:** {model_info['best_model']}")
    st.write(f"**Best hyperparameters:** {model_info['best_params']}")

    report_path = os.path.join(PROJECT_ROOT, "report", "model_performance.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            performance = json.load(f)
        st.write("**Model performance comparison:**")
        perf_df = pd.DataFrame(
            {
                name: {k: v for k, v in res.items() if k != "confusion_matrix"}
                for name, res in performance.items()
            }
        ).T
        st.dataframe(perf_df.style.format("{:.3f}"))

    st.write(
        "This app uses a Logistic Regression model trained on the IBM HR Analytics "
        "Employee Attrition dataset to estimate the probability that an employee "
        "will leave the company, based on demographic, compensation, and "
        "satisfaction-related attributes."
    )
