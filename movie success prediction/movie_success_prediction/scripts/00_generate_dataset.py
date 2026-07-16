"""
Dataset Generation Script
--------------------------
Generates a synthetic movie dataset used for the Movie Success Prediction
project. The dataset simulates realistic relationships between movie
attributes (genre, budget, runtime, cast popularity, release year,
audience ratings, etc.) and commercial success.

Run this script first, before any other script in the pipeline.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# File-anchored paths (work on any machine, any working directory)
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(DATA_DIR, "movies_dataset.csv")

RANDOM_SEED = 42
N_MOVIES = 1500

rng = np.random.default_rng(RANDOM_SEED)

GENRES = [
    "Action", "Comedy", "Drama", "Horror", "Romance",
    "Sci-Fi", "Thriller", "Animation", "Adventure", "Fantasy"
]

GENRE_SUCCESS_BIAS = {
    "Action": 0.10, "Comedy": 0.02, "Drama": -0.02, "Horror": 0.05,
    "Romance": -0.05, "Sci-Fi": 0.08, "Thriller": 0.03, "Animation": 0.12,
    "Adventure": 0.09, "Fantasy": 0.04
}

DIRECTORS_TIER = ["A-List", "Established", "Emerging", "First-Time"]
TIER_BIAS = {"A-List": 0.20, "Established": 0.08, "Emerging": -0.02, "First-Time": -0.10}

STUDIOS = ["Major Studio", "Independent", "Mini-Major"]
STUDIO_BIAS = {"Major Studio": 0.12, "Mini-Major": 0.03, "Independent": -0.08}


def generate_dataset(n=N_MOVIES):
    records = []
    for i in range(n):
        genre = rng.choice(GENRES)
        director_tier = rng.choice(DIRECTORS_TIER, p=[0.12, 0.28, 0.35, 0.25])
        studio = rng.choice(STUDIOS, p=[0.35, 0.35, 0.30])

        release_year = int(rng.integers(2005, 2026))

        # Budget in millions USD, log-normal-ish distribution
        base_budget = rng.lognormal(mean=3.0, sigma=1.0)
        budget_million = float(np.clip(base_budget, 0.5, 300))

        runtime_minutes = int(np.clip(rng.normal(112, 18), 75, 195))

        cast_popularity = float(np.clip(rng.normal(55, 20), 1, 100))
        director_popularity = float(np.clip(rng.normal(50, 22), 1, 100))

        marketing_spend_million = float(np.clip(budget_million * rng.uniform(0.15, 0.55), 0.1, 150))

        num_screens = int(np.clip(rng.normal(2200, 900), 50, 4700))

        critic_score = float(np.clip(rng.normal(58, 18), 1, 100))
        social_media_buzz = float(np.clip(rng.normal(50, 25), 0, 100))

        # Underlying "quality/appeal" score drives audience rating and revenue
        appeal_score = (
            0.25 * (cast_popularity / 100)
            + 0.20 * (director_popularity / 100)
            + 0.20 * (critic_score / 100)
            + 0.15 * (social_media_buzz / 100)
            + 0.10 * (marketing_spend_million / 150)
            + GENRE_SUCCESS_BIAS[genre]
            + TIER_BIAS[director_tier]
            + STUDIO_BIAS[studio]
            + rng.normal(0, 0.08)
        )
        appeal_score = float(np.clip(appeal_score, 0, 1.3))

        audience_rating = float(np.clip(3.5 + appeal_score * 5.5 + rng.normal(0, 0.6), 1.0, 10.0))

        # Revenue driven by budget, marketing, screens and appeal
        revenue_million = (
            budget_million * (1.2 + appeal_score * 3.2)
            + marketing_spend_million * 0.8
            + (num_screens / 1000) * 1.5
            + rng.normal(0, budget_million * 0.35)
        )
        revenue_million = float(max(revenue_million, 0.05))

        roi = (revenue_million - budget_million) / budget_million

        # Success definition: revenue at least 2x budget AND audience_rating >= 6.0
        is_success = int((revenue_million >= 2.0 * budget_million) and (audience_rating >= 6.0))

        records.append({
            "movie_id": i + 1,
            "genre": genre,
            "release_year": release_year,
            "budget_million": round(budget_million, 2),
            "marketing_spend_million": round(marketing_spend_million, 2),
            "runtime_minutes": runtime_minutes,
            "cast_popularity": round(cast_popularity, 1),
            "director_popularity": round(director_popularity, 1),
            "director_tier": director_tier,
            "studio_type": studio,
            "num_screens": num_screens,
            "critic_score": round(critic_score, 1),
            "social_media_buzz": round(social_media_buzz, 1),
            "audience_rating": round(audience_rating, 2),
            "revenue_million": round(revenue_million, 2),
            "roi": round(roi, 2),
            "is_success": is_success
        })

    df = pd.DataFrame(records)

    # Inject a small amount of missing values to simulate real-world data
    for col in ["critic_score", "social_media_buzz", "director_popularity"]:
        missing_idx = rng.choice(df.index, size=int(0.02 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan

    # Inject a handful of duplicate rows
    dup_rows = df.sample(n=15, random_state=RANDOM_SEED)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset generated with {len(dataset)} rows -> {OUTPUT_PATH}")
    print(f"Success rate: {dataset['is_success'].mean():.2%}")
