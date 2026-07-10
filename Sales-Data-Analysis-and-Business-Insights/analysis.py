"""
Sales Data Analysis and Business Insights Report
Data Science Task #02
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# ============================================================
# 1. LOAD AND EXPLORE THE DATASET
# ============================================================
df = pd.read_csv("sales_data_raw.csv")

print("=" * 60)
print("STEP 1: INITIAL EXPLORATION")
print("=" * 60)
print(f"\nShape: {df.shape}")
print(f"\nFull dataset:\n{df.to_string()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nSummary statistics:\n{df.describe()}")

# ============================================================
# 2. DATA CLEANING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: DATA CLEANING")
print("=" * 60)

# Remove exact duplicates
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate row(s)")

# Standardize Region text (fix case inconsistencies)
df["Region"] = df["Region"].str.title()

# Fix negative quantities (data entry errors -> convert to positive)
neg_count = (df["Quantity"] < 0).sum()
df["Quantity"] = df["Quantity"].abs()
print(f"Fixed {neg_count} negative quantity value(s)")

# Handle missing values
df["Quantity"] = df["Quantity"].fillna(df["Quantity"].median())
df["UnitPrice"] = df.groupby("Product")["UnitPrice"].transform(
    lambda x: x.fillna(x.median())
)
df["UnitPrice"] = df["UnitPrice"].fillna(df["UnitPrice"].median())

# Recompute Revenue (Quantity * UnitPrice) for consistency
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# Region: fill missing with mode
df["Region"] = df["Region"].fillna(df["Region"].mode()[0])

# Convert Date to datetime, add Month column
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.strftime("%b")
df["MonthNum"] = df["Date"].dt.month

print(f"\nMissing values after cleaning: {df.isnull().sum().sum()}")
print(f"Final shape: {df.shape}")
print(f"\nCleaned dataset:\n{df.to_string()}")

df.to_csv("sales_data_cleaned.csv", index=False)
print("\nCleaned dataset saved as 'sales_data_cleaned.csv'")

# ============================================================
# 3. OVERALL SALES & REVENUE TRENDS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: OVERALL SALES & REVENUE TRENDS")
print("=" * 60)

total_revenue = df["Revenue"].sum()
total_orders = df["OrderID"].nunique()
total_units = df["Quantity"].sum()
avg_order_value = total_revenue / total_orders

print(f"Total Revenue: PKR {total_revenue:,.2f}")
print(f"Total Orders: {total_orders}")
print(f"Total Units Sold: {total_units:,.0f}")
print(f"Average Order Value: PKR {avg_order_value:,.2f}")

monthly_revenue = df.groupby(["MonthNum", "Month"])["Revenue"].sum().reset_index()
monthly_revenue = monthly_revenue.sort_values("MonthNum")
print(f"\nMonthly Revenue:\n{monthly_revenue[['Month', 'Revenue']]}")

plt.figure()
plt.plot(monthly_revenue["Month"], monthly_revenue["Revenue"], marker="o", color="#2E75B6")
plt.title("Monthly Revenue Trend (2024)", fontsize=14, fontweight="bold")
plt.xlabel("Month")
plt.ylabel("Revenue (PKR)")
plt.tight_layout()
plt.savefig("chart1_monthly_revenue_trend.png", dpi=150)
plt.close()

# ============================================================
# 4. PRODUCT-WISE PERFORMANCE
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: PRODUCT-WISE PERFORMANCE")
print("=" * 60)

product_perf = df.groupby("Product").agg(
    Total_Revenue=("Revenue", "sum"),
    Total_Quantity=("Quantity", "sum"),
    Orders=("OrderID", "count")
).sort_values("Total_Revenue", ascending=False)

print(product_perf)

plt.figure()
product_perf["Total_Revenue"].plot(kind="bar", color="#4472C4")
plt.title("Total Revenue by Product", fontsize=14, fontweight="bold")
plt.xlabel("Product")
plt.ylabel("Revenue (PKR)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("chart2_revenue_by_product.png", dpi=150)
plt.close()

# ============================================================
# 5. BEST AND LEAST SELLING PRODUCTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: BEST & LEAST SELLING PRODUCTS")
print("=" * 60)

best_by_revenue = product_perf["Total_Revenue"].idxmax()
least_by_revenue = product_perf["Total_Revenue"].idxmin()
best_by_qty = product_perf["Total_Quantity"].idxmax()
least_by_qty = product_perf["Total_Quantity"].idxmin()

print(f"Best-selling product (by revenue): {best_by_revenue} "
      f"(PKR {product_perf.loc[best_by_revenue, 'Total_Revenue']:,.2f})")
print(f"Least-selling product (by revenue): {least_by_revenue} "
      f"(PKR {product_perf.loc[least_by_revenue, 'Total_Revenue']:,.2f})")
print(f"Best-selling product (by units): {best_by_qty} "
      f"({product_perf.loc[best_by_qty, 'Total_Quantity']:,.0f} units)")
print(f"Least-selling product (by units): {least_by_qty} "
      f"({product_perf.loc[least_by_qty, 'Total_Quantity']:,.0f} units)")

# ============================================================
# 6. CATEGORY-WISE COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: CATEGORY-WISE COMPARISON")
print("=" * 60)

category_perf = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
print(category_perf)

plt.figure()
plt.pie(category_perf, labels=category_perf.index, autopct="%1.1f%%",
        colors=sns.color_palette("Set2"), startangle=140)
plt.title("Revenue Share by Category", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("chart3_category_share.png", dpi=150)
plt.close()

# ============================================================
# 7. REGIONAL SALES COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: REGIONAL SALES COMPARISON")
print("=" * 60)

region_perf = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
print(region_perf)

plt.figure()
region_perf.plot(kind="bar", color="#70AD47")
plt.title("Total Revenue by Region", fontsize=14, fontweight="bold")
plt.xlabel("Region")
plt.ylabel("Revenue (PKR)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("chart4_revenue_by_region.png", dpi=150)
plt.close()

print("\nAll analysis complete. Charts saved as PNG files.")
