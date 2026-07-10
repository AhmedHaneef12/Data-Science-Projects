===================================================================
TASK 05 – MOVIE RATING ANALYSIS
Gexton Education Summer Internship Program
Supervisor: Sir Muhammad Arham MH
===================================================================

PROJECT OVERVIEW
-----------------
A streaming platform wants to understand which types of movies
perform best based on ratings and user reviews. This project
delivers a complete data science pipeline — data cleaning,
exploratory analysis, and visualisation — to generate insights.


PROJECT STRUCTURE
-----------------
task05_movie_rating_analysis/
├── data/
│   ├── movies_raw.csv           Original dataset (26 records)
│   └── movies_cleaned.csv       Cleaned dataset (23 records)
├── notebooks/
│   └── movie_rating_analysis.ipynb   Jupyter Notebook
├── scripts/
│   └── movie_analysis.py        Python pipeline script
├── charts/
│   ├── chart1_genre_avg_rating.png
│   ├── chart2_top5_movies.png
│   └── chart3_rating_distribution.png
├── reports/
│   └── Task05_Movie_Rating_Analysis_Report.docx
├── README.md
└── README.txt


DATASET COLUMNS
---------------
Movie Name       | string  | Title of the movie
Genre            | string  | Genre category
Release Year     | int     | Year of release
Rating           | float   | Audience rating (1–10)
Number of Reviews| int     | Total review count
Budget (USD)     | int     | Production budget (0 = unknown)

Raw dataset  : 26 records (1 duplicate, 2 missing ratings, 4 missing budgets)
Cleaned dataset : 23 records across 7 genres (1984–2019)


PIPELINE
--------
1. DATA CLEANING
   - Removed 1 duplicate row (Gone Girl)
   - Dropped 2 rows with missing Rating (required field)
   - Filled 4 missing Budget values with 0 (optional field)
   - Enforced correct data types for all numeric columns

2. BASIC ANALYSIS
   - Average rating       : 8.11 / 10
   - Highest-rated movie  : The Shawshank Redemption (9.3)
   - Best genre           : Drama (avg 8.80)
   - Most-reviewed        : The Shawshank Redemption (2.8M reviews)

3. VISUALISATION
   - Bar chart         - Genre vs Average Rating
   - Horizontal bar    - Top 5 Movies by Rating
   - Histogram         - Rating Distribution

4. INSIGHTS
   - Drama is the top-performing genre (avg 8.80)
   - High-budget films (>$100M) average 8.40 vs 8.06 for lower-budget
   - Ratings cluster tightly between 7.5-9.0 (mean 8.11, std 0.57)


HOW TO RUN
----------
  python scripts/movie_analysis.py

Requirements: Python 3, pandas, numpy, matplotlib, seaborn


TOOLS USED
----------
Python 3 | Pandas | NumPy | Matplotlib | Seaborn | VS Code

===================================================================
