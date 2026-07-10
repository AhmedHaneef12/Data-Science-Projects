import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110

# 1. Load data
df = pd.read_csv('/mnt/user-data/uploads/global_ecommerce_sales.csv')
print("Initial shape:", df.shape)

# 2. Cleaning
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
before = df.shape[0]
df = df.drop_duplicates()
missing = df.isnull().sum().sum()
df = df.dropna()
print(f"Duplicates removed: {before - df.shape[0]}, Missing values found: {missing}")

# Feature engineering - date parts
df['Year'] = df['Order_Date'].dt.year
df['Month'] = df['Order_Date'].dt.month
df['MonthName'] = df['Order_Date'].dt.strftime('%b %Y')
df['DayOfWeek'] = df['Order_Date'].dt.day_name()
df['Quarter'] = df['Order_Date'].dt.quarter

# Net Revenue after discount check - Total_Sales already reflects discount? Let's verify
df['Gross_Amount'] = df['Quantity'] * df['Unit_Price']
df['Expected_Sales_After_Discount'] = df['Gross_Amount'] * (1 - df['Discount_Percent']/100)
df['Profit_Margin_Pct'] = (df['Profit'] / df['Total_Sales'] * 100).round(2)

df.to_csv('/home/claude/ecommerce_project/data/global_ecommerce_sales_cleaned.csv', index=False)
print("Cleaned shape:", df.shape)
print(df[['Gross_Amount','Expected_Sales_After_Discount','Total_Sales']].head())
