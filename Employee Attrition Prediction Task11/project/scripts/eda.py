"""
Exploratory Data Analysis Script
Project: Employee Attrition Prediction & HR Analytics

Generates visualizations of employee demographics and attrition patterns,
including a correlation heatmap and outlier detection, and saves all
charts to the charts/ folder.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Anchor all paths to this script's location so the script works the same way
# whether it's run from a terminal, VS Code's "Run" button, or another machine.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_attrition_data.csv")
CHARTS_DIR = os.path.join(PROJECT_ROOT, "charts")

sns.set_theme(style="whitegrid")


def save_fig(fig, name):
    fig.savefig(os.path.join(CHARTS_DIR, name), bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_attrition_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Attrition", palette=["#4C72B0", "#DD8452"], ax=ax)
    ax.set_title("Overall Attrition Distribution")
    save_fig(fig, "01_attrition_distribution.png")


def plot_age_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="Age", hue="Attrition", kde=True, multiple="stack", ax=ax)
    ax.set_title("Age Distribution by Attrition")
    save_fig(fig, "02_age_distribution.png")


def plot_attrition_by_department(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="Department", hue="Attrition", ax=ax)
    ax.set_title("Attrition by Department")
    plt.xticks(rotation=15)
    save_fig(fig, "03_attrition_by_department.png")


def plot_attrition_by_jobrole(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    order = df["JobRole"].value_counts().index
    sns.countplot(data=df, y="JobRole", hue="Attrition", order=order, ax=ax)
    ax.set_title("Attrition by Job Role")
    save_fig(fig, "04_attrition_by_jobrole.png")


def plot_attrition_by_overtime(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="OverTime", hue="Attrition", ax=ax)
    ax.set_title("Attrition by OverTime Status")
    save_fig(fig, "05_attrition_by_overtime.png")


def plot_income_vs_attrition(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", ax=ax)
    ax.set_title("Monthly Income vs Attrition (Outlier Detection)")
    save_fig(fig, "06_income_vs_attrition_boxplot.png")


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Features")
    save_fig(fig, "07_correlation_heatmap.png")


def plot_gender_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Gender", hue="Attrition", ax=ax)
    ax.set_title("Attrition by Gender")
    save_fig(fig, "08_attrition_by_gender.png")


def plot_worklife_balance(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="WorkLifeBalance", hue="Attrition", ax=ax)
    ax.set_title("Attrition by Work-Life Balance Rating")
    save_fig(fig, "09_attrition_by_worklifebalance.png")


def plot_distance_from_home(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="DistanceFromHome", hue="Attrition", kde=True, multiple="stack", ax=ax)
    ax.set_title("Distance From Home Distribution by Attrition")
    save_fig(fig, "10_distance_from_home.png")


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    plot_attrition_distribution(df)
    plot_age_distribution(df)
    plot_attrition_by_department(df)
    plot_attrition_by_jobrole(df)
    plot_attrition_by_overtime(df)
    plot_income_vs_attrition(df)
    plot_correlation_heatmap(df)
    plot_gender_distribution(df)
    plot_worklife_balance(df)
    plot_distance_from_home(df)

    print("All charts saved to", CHARTS_DIR)


if __name__ == "__main__":
    main()
