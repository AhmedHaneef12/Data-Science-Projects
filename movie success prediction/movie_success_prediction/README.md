# 🎬 Movie Success Prediction

An end-to-end machine learning project that predicts whether a movie is likely to become a commercial success, based on attributes such as genre, budget, runtime, cast popularity, release year, and audience ratings. Built as Data Science Task #12 for the Gexton Education Summer Internship Program.

---

## 📌 Project Overview

The goal of this project is to analyze a dataset of movie attributes and build a classification model that predicts commercial success. The project covers the complete data science lifecycle:

- Data preparation and exploratory data analysis (EDA)
- Machine learning model development and comparison
- Prediction on new movie records
- Business insights and recommendations for producers and studios
- A deployable Streamlit web application

---

## 🗂️ Project Structure

```
movie_success_prediction/
│
├── data/                            # Raw and processed datasets
│   ├── movies_dataset.csv               # Raw dataset
│   ├── movies_dataset_clean.csv         # Cleaned dataset (post-EDA)
│   └── new_movie_predictions.csv        # Predictions on sample new movies
│
├── scripts/                         # Pipeline scripts (run in order)
│   ├── 00_generate_dataset.py           # Generates the movie dataset
│   ├── 01_eda.py                        # Cleaning + exploratory analysis + charts
│   ├── 02_train_models.py               # Model training, tuning, evaluation
│   └── 03_predict_and_insights.py       # Predictions + business insights
│
├── notebooks/
│   └── movie_success_prediction.ipynb   # Full analysis in notebook form
│
├── models/
│   ├── best_model.pkl                   # Saved best-performing model (joblib)
│   ├── model_comparison.csv             # Metrics for all trained models
│   └── metrics.json                     # Summary metrics of the best model
│
├── charts/                          # All generated visualizations (PNG)
│
├── app/
│   └── app.py                           # Streamlit web application
│
├── report/
│   ├── Movie_Success_Prediction_Report.docx   # Full written report
│   ├── Movie_Success_Prediction_Presentation.pptx  # Presentation slides
│   └── business_insights.txt            # Business insights (plain text)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

The dataset consists of **1,500 movie records** with the following features:

| Feature | Description |
|---|---|
| `genre` | Primary genre of the movie |
| `release_year` | Year the movie was released |
| `budget_million` | Production budget (USD, millions) |
| `marketing_spend_million` | Marketing spend (USD, millions) |
| `runtime_minutes` | Runtime in minutes |
| `cast_popularity` | Cast popularity score (0–100) |
| `director_popularity` | Director popularity score (0–100) |
| `director_tier` | Director experience tier (A-List, Established, Emerging, First-Time) |
| `studio_type` | Studio type (Major Studio, Mini-Major, Independent) |
| `num_screens` | Number of theatrical screens at release |
| `critic_score` | Critic score (0–100) |
| `social_media_buzz` | Pre-release social media buzz score (0–100) |
| `audience_rating` | Audience rating (0–10) |
| `revenue_million` | Box office revenue (USD, millions) |
| `roi` | Return on investment |
| `is_success` | **Target** — 1 if successful, 0 otherwise |

**Success definition:** a movie is labeled successful if its revenue is at least 2x its budget **and** its audience rating is 6.0 or higher.

> **Note:** `audience_rating` is excluded from the model's feature set during training, since it directly informed the target definition. Including it would leak target information into the model.

---

## 🤖 Machine Learning Approach

Three classification models were trained and tuned with `GridSearchCV` (5-fold cross-validation, F1-score optimization):

- Logistic Regression
- Random Forest
- Decision Tree

Each model was evaluated on Accuracy, Precision, Recall, and F1-Score. The best-performing model was selected and saved with `joblib` for reuse in the Streamlit application and for making predictions on new movie records.

Feature importance analysis was used to identify which factors most influence movie success — see `charts/feature_importance.png` and `report/business_insights.txt` for details.

---

## 🚀 Getting Started (Local Setup — VS Code)

All scripts use **file-anchored relative paths** (`Path(__file__)`), so the project runs correctly from any location on any machine — no manual path edits required.

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd movie_success_prediction
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline (in order)

```bash
python scripts/00_generate_dataset.py
python scripts/01_eda.py
python scripts/02_train_models.py
python scripts/03_predict_and_insights.py
```

This will regenerate the dataset, charts, trained model, and prediction/insight outputs from scratch.

### 5. Launch the Streamlit app

```bash
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`, with tabs for project overview, EDA, model performance, live prediction, and business insights.

### 6. Explore the notebook (optional)

Open `notebooks/movie_success_prediction.ipynb` in VS Code or Jupyter to walk through the full analysis interactively.

---

## 📈 Results Summary

The best-performing model and its evaluation metrics are saved in `models/metrics.json` and `models/model_comparison.csv` after running the training script. Full details, charts, and discussion are available in the Word report (`report/Movie_Success_Prediction_Report.docx`) and presentation (`report/Movie_Success_Prediction_Presentation.pptx`).

---

## 💡 Key Business Insights

- Genre, director experience, and studio type all show measurable effects on success rate.
- Wider theatrical distribution (screen count) and marketing spend are positively associated with commercial success.
- Critic score and pre-release social media buzz act as useful leading indicators of performance.

Full recommendations are available in `report/business_insights.txt`.

---

## 🛠️ Tech Stack

- **Python** — pandas, NumPy, scikit-learn, joblib
- **Visualization** — Matplotlib, Seaborn
- **Web App** — Streamlit
- **Environment** — VS Code, Jupyter Notebook

---

## 📄 License

This project was developed for educational purposes as part of the Gexton Education Summer Internship Program.
