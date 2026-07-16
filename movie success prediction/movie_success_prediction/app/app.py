"""
Movie Success Prediction — Streamlit Application
----------------------------------------------------
A multi-tab Streamlit app for exploring the movie dataset, reviewing
model performance, and predicting the success of new movie records.

Run locally with:
    streamlit run app/app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# File-anchored paths (work on any machine, any working directory)
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")

CLEAN_DATA_PATH = os.path.join(DATA_DIR, "movies_dataset_clean.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.csv")

st.set_page_config(page_title="Movie Success Prediction", page_icon="🎬", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(CLEAN_DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def check_prerequisites():
    missing = []
    if not os.path.exists(CLEAN_DATA_PATH):
        missing.append("data/movies_dataset_clean.csv (run scripts/01_eda.py)")
    if not os.path.exists(MODEL_PATH):
        missing.append("models/best_model.pkl (run scripts/02_train_models.py)")
    if missing:
        st.error(
            "Required project files are missing. Please run the pipeline scripts "
            "in order before launching the app:\n\n" + "\n".join(f"- {m}" for m in missing)
        )
        st.stop()


check_prerequisites()
df = load_data()
model = load_model()

st.title("🎬 Movie Success Prediction")
st.caption("An end-to-end machine learning project analyzing and predicting commercial movie success.")

tab_overview, tab_eda, tab_model, tab_predict, tab_insights = st.tabs(
    ["📌 Overview", "📊 Exploratory Analysis", "🤖 Model Performance", "🎯 Predict a Movie", "💡 Business Insights"]
)

# ---------------------------------------------------------------
# TAB 1: Overview
# ---------------------------------------------------------------
with tab_overview:
    st.subheader("Project Overview")
    st.write(
        "This project builds a machine learning model to predict whether a movie is "
        "likely to become a commercial success, based on attributes such as genre, "
        "budget, runtime, cast popularity, release year, and audience ratings."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Movies", len(df))
    col2.metric("Success Rate", f"{df['is_success'].mean():.1%}")
    col3.metric("Genres Covered", df["genre"].nunique())
    col4.metric("Year Range", f"{df['release_year'].min()}–{df['release_year'].max()}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Dataset Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

# ---------------------------------------------------------------
# TAB 2: EDA
# ---------------------------------------------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis")
    st.write("Key visual trends discovered during data exploration.")

    chart_files = {
        "Genre Distribution": "genre_distribution.png",
        "Budget vs. Revenue": "budget_vs_revenue.png",
        "Audience Rating Distribution": "rating_distribution.png",
        "Runtime vs. Success": "runtime_vs_success.png",
        "Success Rate by Genre": "success_rate_by_genre.png",
        "Correlation Heatmap": "correlation_heatmap.png",
        "Success by Studio & Director": "success_by_studio_and_director.png",
    }

    col1, col2 = st.columns(2)
    for i, (title, filename) in enumerate(chart_files.items()):
        path = os.path.join(CHARTS_DIR, filename)
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            st.markdown(f"**{title}**")
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.warning(f"Chart not found: {filename}. Run scripts/01_eda.py first.")

    st.subheader("Explore the Data Yourself")
    genre_filter = st.multiselect("Filter by genre", options=sorted(df["genre"].unique()),
                                   default=sorted(df["genre"].unique()))
    filtered_df = df[df["genre"].isin(genre_filter)]

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.scatterplot(data=filtered_df, x="critic_score", y="audience_rating",
                     hue="is_success", palette={0: "#e74c3c", 1: "#2ecc71"}, alpha=0.6, ax=ax)
    ax.set_title("Critic Score vs. Audience Rating")
    st.pyplot(fig)

# ---------------------------------------------------------------
# TAB 3: Model Performance
# ---------------------------------------------------------------
with tab_model:
    st.subheader("Model Comparison")
    if os.path.exists(COMPARISON_PATH):
        comparison_df = pd.read_csv(COMPARISON_PATH)
        st.dataframe(
            comparison_df[["model", "accuracy", "precision", "recall", "f1_score"]],
            use_container_width=True
        )
    else:
        st.warning("Model comparison file not found. Run scripts/02_train_models.py first.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Model Performance Comparison**")
        model_comp_chart = os.path.join(CHARTS_DIR, "model_comparison.png")
        if os.path.exists(model_comp_chart):
            st.image(model_comp_chart, use_container_width=True)

        st.markdown("**Confusion Matrix (Best Model)**")
        cm_chart = os.path.join(CHARTS_DIR, "confusion_matrix.png")
        if os.path.exists(cm_chart):
            st.image(cm_chart, use_container_width=True)

    with col2:
        st.markdown("**Feature Importance**")
        fi_chart = os.path.join(CHARTS_DIR, "feature_importance.png")
        if os.path.exists(fi_chart):
            st.image(fi_chart, use_container_width=True)

# ---------------------------------------------------------------
# TAB 4: Predict a Movie
# ---------------------------------------------------------------
with tab_predict:
    st.subheader("Predict the Success of a New Movie")
    st.write("Enter movie attributes below to get a prediction from the trained model.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            genre = st.selectbox("Genre", sorted(df["genre"].unique()))
            release_year = st.number_input("Release Year", min_value=2000, max_value=2035, value=2026)
            budget_million = st.number_input("Budget (Million USD)", min_value=0.1, value=50.0, step=1.0)
            marketing_spend_million = st.number_input("Marketing Spend (Million USD)", min_value=0.0, value=15.0, step=1.0)

        with col2:
            runtime_minutes = st.number_input("Runtime (Minutes)", min_value=60, max_value=240, value=115)
            cast_popularity = st.slider("Cast Popularity (0-100)", 0, 100, 60)
            director_popularity = st.slider("Director Popularity (0-100)", 0, 100, 55)
            director_tier = st.selectbox("Director Tier", ["A-List", "Established", "Emerging", "First-Time"])

        with col3:
            studio_type = st.selectbox("Studio Type", ["Major Studio", "Mini-Major", "Independent"])
            num_screens = st.number_input("Number of Screens", min_value=1, max_value=6000, value=2500)
            critic_score = st.slider("Critic Score (0-100)", 0, 100, 60)
            social_media_buzz = st.slider("Social Media Buzz (0-100)", 0, 100, 50)

        submitted = st.form_submit_button("Predict Success")

    if submitted:
        input_df = pd.DataFrame([{
            "genre": genre,
            "release_year": release_year,
            "budget_million": budget_million,
            "marketing_spend_million": marketing_spend_million,
            "runtime_minutes": runtime_minutes,
            "cast_popularity": cast_popularity,
            "director_popularity": director_popularity,
            "director_tier": director_tier,
            "studio_type": studio_type,
            "num_screens": num_screens,
            "critic_score": critic_score,
            "social_media_buzz": social_media_buzz
        }])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0, 1]

        if prediction == 1:
            st.success(f"✅ Predicted: **Successful** (probability: {probability:.1%})")
        else:
            st.error(f"❌ Predicted: **Not Successful** (probability of success: {probability:.1%})")

        st.progress(min(max(probability, 0.0), 1.0))

# ---------------------------------------------------------------
# TAB 5: Business Insights
# ---------------------------------------------------------------
with tab_insights:
    st.subheader("Business Insights & Recommendations")
    insights_path = os.path.join(BASE_DIR, "report", "business_insights.txt")
    if os.path.exists(insights_path):
        with open(insights_path, "r") as f:
            st.text(f.read())
    else:
        st.warning("Business insights file not found. Run scripts/03_predict_and_insights.py first.")

st.divider()
st.caption("Movie Success Prediction — Gexton Education Summer Internship Program, Data Science Task #12")
