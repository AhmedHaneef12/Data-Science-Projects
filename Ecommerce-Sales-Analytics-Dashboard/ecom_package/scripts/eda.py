import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110
palette = sns.color_palette('viridis', 10)

df = pd.read_csv('data/global_ecommerce_sales_cleaned.csv', parse_dates=['Order_Date'])

# 1. Monthly Sales Trend
monthly = df.groupby(df['Order_Date'].dt.to_period('M')).agg(Total_Sales=('Total_Sales','sum'), Orders=('Order_ID','count')).reset_index()
monthly['Order_Date'] = monthly['Order_Date'].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(11,4.5))
ax.plot(monthly['Order_Date'], monthly['Total_Sales'], marker='o', color='#2a6f97', linewidth=2)
ax.set_title('Monthly Sales Trend', fontsize=14, fontweight='bold')
ax.set_xlabel('Month'); ax.set_ylabel('Total Sales ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('images/01_monthly_sales_trend.png')
plt.close()

# 2. Top Selling Products (by revenue)
top_products = df.groupby('Product_Name')['Total_Sales'].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(x=top_products.values, y=top_products.index, hue=top_products.index, palette='viridis', ax=ax, legend=False)
ax.set_title('Top 10 Selling Products by Revenue', fontsize=14, fontweight='bold')
ax.set_xlabel('Total Sales ($)'); ax.set_ylabel('')
plt.tight_layout()
plt.savefig('images/02_top_products.png')
plt.close()

# 3. Best Performing Categories
cat_perf = df.groupby('Product_Category').agg(Total_Sales=('Total_Sales','sum'), Profit=('Profit','sum')).sort_values('Total_Sales', ascending=False)
fig, ax = plt.subplots(figsize=(9,5))
x = np.arange(len(cat_perf))
width=0.38
ax.bar(x-width/2, cat_perf['Total_Sales'], width, label='Total Sales', color='#2a6f97')
ax.bar(x+width/2, cat_perf['Profit'], width, label='Profit', color='#f4a261')
ax.set_xticks(x); ax.set_xticklabels(cat_perf.index, rotation=15)
ax.set_title('Category Performance: Sales vs Profit', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('images/03_category_performance.png')
plt.close()

# 4. Regional Sales Performance
region_perf = df.groupby('Region')['Total_Sales'].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x=region_perf.index, y=region_perf.values, hue=region_perf.index, palette='crest', ax=ax, legend=False)
ax.set_title('Regional Sales Performance', fontsize=14, fontweight='bold')
ax.set_ylabel('Total Sales ($)'); ax.set_xlabel('')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('images/04_regional_performance.png')
plt.close()

# 5. Customer Purchase Behavior (segment)
seg = df.groupby('Customer_Segment').agg(Total_Sales=('Total_Sales','sum'), Avg_Order=('Total_Sales','mean'), Orders=('Order_ID','count'))
fig, axes = plt.subplots(1,2, figsize=(12,5))
sns.barplot(x=seg.index, y=seg['Total_Sales'], hue=seg.index, palette='mako', ax=axes[0], legend=False)
axes[0].set_title('Total Sales by Customer Segment', fontweight='bold')
sns.barplot(x=seg.index, y=seg['Avg_Order'], hue=seg.index, palette='mako', ax=axes[1], legend=False)
axes[1].set_title('Average Order Value by Segment', fontweight='bold')
plt.tight_layout()
plt.savefig('images/05_customer_segment.png')
plt.close()

# 6. Payment Method Analysis
pay = df['Payment_Method'].value_counts()
fig, ax = plt.subplots(figsize=(6.5,6.5))
colors = sns.color_palette('Set2', len(pay))
ax.pie(pay.values, labels=pay.index, autopct='%1.1f%%', colors=colors, startangle=90,
       wedgeprops={'edgecolor':'white','linewidth':1.5})
ax.set_title('Payment Method Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/06_payment_method.png')
plt.close()

# 7. Revenue & Profit Overview (yearly)
yearly = df.groupby('Year').agg(Total_Sales=('Total_Sales','sum'), Profit=('Profit','sum'), Orders=('Order_ID','count'))
fig, ax = plt.subplots(figsize=(8,5))
x = np.arange(len(yearly))
width = 0.38
ax.bar(x-width/2, yearly['Total_Sales'], width, label='Revenue', color='#264653')
ax.bar(x+width/2, yearly['Profit'], width, label='Profit', color='#e76f51')
ax.set_xticks(x); ax.set_xticklabels(yearly.index)
ax.set_title('Yearly Revenue vs Profit', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('images/07_revenue_profit_overview.png')
plt.close()

# 8. Correlation heatmap for numeric features
numeric_cols = ['Quantity','Unit_Price','Discount_Percent','Total_Sales','Shipping_Cost','Profit']
fig, ax = plt.subplots(figsize=(7,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/08_correlation_heatmap.png')
plt.close()

# Print summary stats for report
print("=== KPI SUMMARY ===")
print("Total Revenue:", round(df['Total_Sales'].sum(),2))
print("Total Profit:", round(df['Profit'].sum(),2))
print("Total Orders:", df['Order_ID'].nunique())
print("Avg Order Value:", round(df['Total_Sales'].mean(),2))
print("Overall Profit Margin %:", round(df['Profit'].sum()/df['Total_Sales'].sum()*100,2))
print()
print("Top Category:", cat_perf.index[0])
print("Top Region:", region_perf.index[0])
print("Top Product:", top_products.index[0])
print("Top Payment Method:", pay.index[0], f"({pay.values[0]} orders, {pay.values[0]/pay.sum()*100:.1f}%)")
print()
print("Segment breakdown:\n", seg)
print()
print("Yearly breakdown:\n", yearly)
