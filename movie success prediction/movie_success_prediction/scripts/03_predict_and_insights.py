"""
Prediction & Business Insights Script
-----------------------------------------
Loads the saved best model and generates:
    1. Predictions for a batch of new/sample movie records.
    2. A business insights summary (text) based on the trained model
       and EDA findings, useful for producers and production companies.

Outputs:
    data/new_movie_predictions.csv
    report/business_insights.txt
"""

import os
import joblib
import pandas as pd

# ---------------------------------------------------------------
# File-anchored paths
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
CLEAN_PATH = os.path.join(DATA_DIR, "movies_dataset_clean.csv")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "new_movie_predictions.csv")
INSIGHTS_PATH = os.path.join(REPORT_DIR, "business_insights.txt")

# Sample "new" movie records to demonstrate prediction on unseen data
NEW_MOVIES = pd.DataFrame([
    {
        "movie_id": "NEW-001", "genre": "Action", "release_year": 2026,
        "budget_million": 120, "marketing_spend_million": 45,
        "runtime_minutes": 128, "cast_popularity": 82, "director_popularity": 74,
        "director_tier": "A-List", "studio_type": "Major Studio",
        "num_screens": 4000, "critic_score": 68, "social_media_buzz": 77
    },
    {
        "movie_id": "NEW-002", "genre": "Drama", "release_year": 2026,
        "budget_million": 8, "marketing_spend_million": 2,
        "runtime_minutes": 105, "cast_popularity": 38, "director_popularity": 41,
        "director_tier": "Emerging", "studio_type": "Independent",
        "num_screens": 350, "critic_score": 72, "social_media_buzz": 30
    },
    {
        "movie_id": "NEW-003", "genre": "Animation", "release_year": 2026,
        "budget_million": 95, "marketing_spend_million": 40,
        "runtime_minutes": 98, "cast_popularity": 60, "director_popularity": 65,
        "director_tier": "Established", "studio_type": "Major Studio",
        "num_screens": 3800, "critic_score": 75, "social_media_buzz": 82
    },
    {
        "movie_id": "NEW-004", "genre": "Horror", "release_year": 2026,
        "budget_million": 12, "marketing_spend_million": 6,
        "runtime_minutes": 92, "cast_popularity": 30, "director_popularity": 28,
        "director_tier": "First-Time", "studio_type": "Independent",
        "num_screens": 800, "critic_score": 45, "social_media_buzz": 55
    },
    {
        "movie_id": "NEW-005", "genre": "Sci-Fi", "release_year": 2026,
        "budget_million": 180, "marketing_spend_million": 70,
        "runtime_minutes": 140, "cast_popularity": 88, "director_popularity": 80,
        "director_tier": "A-List", "studio_type": "Major Studio",
        "num_screens": 4500, "critic_score": 80, "social_media_buzz": 90
    },
])


def predict_new_movies(model):
    feature_cols = [c for c in NEW_MOVIES.columns if c != "movie_id"]
    predictions = model.predict(NEW_MOVIES[feature_cols])
    probabilities = model.predict_proba(NEW_MOVIES[feature_cols])[:, 1]

    result = NEW_MOVIES.copy()
    result["predicted_success"] = predictions
    result["success_probability"] = probabilities.round(3)
    result["predicted_success_label"] = result["predicted_success"].map(
        {1: "Successful", 0: "Not Successful"}
    )
    return result


def generate_insights(df):
    genre_rate = df.groupby("genre")["is_success"].mean().sort_values(ascending=False)
    top_genres = genre_rate.head(3)
    bottom_genres = genre_rate.tail(3)

    tier_rate = df.groupby("director_tier")["is_success"].mean().sort_values(ascending=False)
    studio_rate = df.groupby("studio_type")["is_success"].mean().sort_values(ascending=False)

    avg_budget_success = df.loc[df["is_success"] == 1, "budget_million"].mean()
    avg_budget_fail = df.loc[df["is_success"] == 0, "budget_million"].mean()

    avg_roi_success = df.loc[df["is_success"] == 1, "roi"].mean()

    corr_screens = df["num_screens"].corr(df["is_success"])
    corr_marketing = df["marketing_spend_million"].corr(df["is_success"])
    corr_critic = df["critic_score"].corr(df["is_success"])

    lines = []
    lines.append("MOVIE SUCCESS PREDICTION — BUSINESS INSIGHTS & RECOMMENDATIONS")
    lines.append("=" * 65)
    lines.append("")
    lines.append("1. Genre Performance")
    lines.append("-" * 65)
    lines.append(f"Top performing genres by success rate: {', '.join(top_genres.index)}")
    for g, v in top_genres.items():
        lines.append(f"   - {g}: {v:.1%} success rate")
    lines.append(f"Lowest performing genres: {', '.join(bottom_genres.index)}")
    for g, v in bottom_genres.items():
        lines.append(f"   - {g}: {v:.1%} success rate")
    lines.append("")
    lines.append("2. Director & Studio Effects")
    lines.append("-" * 65)
    lines.append("Success rate by director tier:")
    for t, v in tier_rate.items():
        lines.append(f"   - {t}: {v:.1%}")
    lines.append("Success rate by studio type:")
    for s, v in studio_rate.items():
        lines.append(f"   - {s}: {v:.1%}")
    lines.append("")
    lines.append("3. Budget & ROI Patterns")
    lines.append("-" * 65)
    lines.append(f"Average budget of successful movies: ${avg_budget_success:.1f}M")
    lines.append(f"Average budget of unsuccessful movies: ${avg_budget_fail:.1f}M")
    lines.append(f"Average ROI of successful movies: {avg_roi_success:.2f}x")
    lines.append("")
    lines.append("4. Distribution & Marketing Correlations")
    lines.append("-" * 65)
    lines.append(f"Correlation between screen count and success: {corr_screens:.2f}")
    lines.append(f"Correlation between marketing spend and success: {corr_marketing:.2f}")
    lines.append(f"Correlation between critic score and success: {corr_critic:.2f}")
    lines.append("")
    lines.append("5. Recommendations for Producers & Production Companies")
    lines.append("-" * 65)
    lines.append("   - Prioritize genres with consistently higher success rates")
    lines.append("     (Animation, Action, and Sci-Fi in this dataset) when allocating")
    lines.append("     development budget for high-stakes projects.")
    lines.append("   - Partner with A-List or Established directors where possible;")
    lines.append("     director track record shows a measurable lift in success rate.")
    lines.append("   - Wide theatrical distribution (higher screen counts) is positively")
    lines.append("     associated with success — negotiate for stronger screen commitments")
    lines.append("     during distribution planning.")
    lines.append("   - Marketing spend shows a positive relationship with success; ensure")
    lines.append("     marketing budgets scale appropriately with production budget rather")
    lines.append("     than being cut disproportionately.")
    lines.append("   - Critic and early buzz scores are useful leading indicators — track")
    lines.append("     them during pre-release windows to adjust marketing strategy.")
    lines.append("   - Independent studio releases succeed less often on average; smaller")
    lines.append("     studios should focus on strong critic scores and social buzz to")
    lines.append("     offset lower marketing budgets.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(CLEAN_PATH)

    predictions = predict_new_movies(model)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions for new movies saved -> {PREDICTIONS_PATH}")
    print(predictions[["movie_id", "genre", "predicted_success_label", "success_probability"]])

    insights_text = generate_insights(df)
    with open(INSIGHTS_PATH, "w") as f:
        f.write(insights_text)
    print(f"\nBusiness insights saved -> {INSIGHTS_PATH}")
