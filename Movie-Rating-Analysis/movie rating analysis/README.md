# Task 05 – Movie Rating Analysis
**Gexton Education Summer Internship Program**  
**Supervisor:** Sir Muhammad Arham MH

---

## Project Overview
A streaming platform wants to understand which types of movies perform best based on ratings and user reviews. This project delivers a complete data science pipeline — data cleaning, exploratory analysis, and visualisation — to generate actionable insights.

---

## Project Structure
```
task05_movie_rating_analysis/
├── data/
│   ├── movies_raw.csv          # Original dataset (26 records, intentional quality issues)
│   └── movies_cleaned.csv      # Cleaned dataset (23 records, ready for analysis)
├── notebooks/
│   └── movie_rating_analysis.ipynb   # Jupyter Notebook (full pipeline)
├── scripts/
│   └── movie_analysis.py       # Standalone Python pipeline script
├── charts/
│   ├── chart1_genre_avg_rating.png   # Genre vs Average Rating (bar chart)
│   ├── chart2_top5_movies.png        # Top 5 Movies by Rating (horizontal bar)
│   └── chart3_rating_distribution.png # Rating Distribution (histogram)
├── reports/
│   └── Task05_Movie_Rating_Analysis_Report.docx
├── README.md
└── README.txt
```

---

## Dataset
| Column | Type | Description |
|---|---|---|
| Movie Name | string | Title of the movie |
| Genre | string | Genre category |
| Release Year | int | Year of release |
| Rating | float | Audience rating (1–10) |
| Number of Reviews | int | Total review count |
| Budget (USD) | int | Production budget (0 = unknown) |

**Raw:** 26 records — 1 duplicate, 2 missing ratings, 4 missing budgets  
**Cleaned:** 23 records across 7 genres (1984–2019)

---

## Pipeline

### 1. Data Cleaning
- Removed 1 duplicate row (*Gone Girl*)
- Dropped 2 rows with missing `Rating` (required field)
- Filled 4 missing `Budget` values with 0 (optional field, flagged as unknown)
- Enforced correct data types for all numeric columns

### 2. Basic Analysis
- **Average rating:** 8.11 / 10
- **Highest-rated movie:** The Shawshank Redemption (9.3)
- **Best genre:** Drama (avg 8.80)
- **Most-reviewed:** The Shawshank Redemption (2.8M reviews)

### 3. Visualisation
- Bar chart — Genre vs Average Rating
- Horizontal bar chart — Top 5 Movies by Rating
- Histogram — Rating Distribution

---

## Key Insights
1. **Drama** is the top-performing genre (avg 8.80), led by *The Shawshank Redemption* and *Parasite*.
2. High-budget films (>$100M) average 8.40 vs 8.06 for lower-budget ones — a modest gap showing budget is not the primary rating driver.
3. Ratings cluster tightly between 7.5 and 9.0 (mean 8.11, std 0.57), indicating a consistently high-quality collection.

---

## How to Run
```bash
# Run the standalone analysis script
python scripts/movie_analysis.py

# Or open the Jupyter Notebook
jupyter notebook notebooks/movie_rating_analysis.ipynb
```

**Requirements:** Python 3, pandas, numpy, matplotlib, seaborn

---

## Tools Used
Python 3 · Pandas · NumPy · Matplotlib · Seaborn · VS Code
