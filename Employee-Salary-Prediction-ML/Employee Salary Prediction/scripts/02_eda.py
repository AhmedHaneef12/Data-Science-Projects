"""
Task #06 - Employee Salary Prediction
Script 02: Exploratory Data Analysis (EDA)
Dataset : Employee Salary Dataset — Kaggle
Source  : https://www.kaggle.com/datasets/prince7489/employee-salary-dataset/data
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import gaussian_kde
from pathlib import Path

BASE   = Path(__file__).parent.parent
CLEAN  = BASE / 'data' / 'cleaned_employee_salary.csv'
CHARTS = BASE / 'charts'

# ── Global Style ─────────────────────────────────────────────────────────────
PRIMARY  = '#1E3A5F'
ACCENT   = '#E07B39'
GREEN    = '#27AE60'
RED      = '#E74C3C'
LIGHT_BG = '#F7F9FC'
GRID_CLR = '#DDE4ED'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'axes.labelsize': 12, 'axes.labelcolor': PRIMARY,
    'axes.edgecolor': PRIMARY, 'axes.facecolor': LIGHT_BG,
    'figure.facecolor': 'white', 'xtick.color': PRIMARY,
    'ytick.color': PRIMARY, 'grid.color': GRID_CLR,
    'grid.linewidth': 0.8, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

df = pd.read_csv(CLEAN)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

edu_order   = ['High School', 'Bachelor', 'Master', 'PhD']
edu_palette = {'High School': RED, 'Bachelor': '#3498DB', 'Master': GREEN, 'PhD': '#9B59B6'}

def fmt_inr(x, _): return f'₹{x/1000:.0f}K'

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Experience vs Monthly Salary (scatter, coloured by Education)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
for edu in edu_order:
    mask = df['Education_Level'] == edu
    ax.scatter(df.loc[mask, 'Experience_Years'], df.loc[mask, 'Monthly_Salary'],
               color=edu_palette[edu], s=80, alpha=0.80, label=edu,
               edgecolors='white', linewidths=0.6, zorder=3)

z  = np.polyfit(df['Experience_Years'], df['Monthly_Salary'], 1)
xs = np.linspace(0, df['Experience_Years'].max(), 100)
ax.plot(xs, np.poly1d(z)(xs), color=ACCENT, lw=2.5, ls='--', label='Trend Line', zorder=4)

r = df['Experience_Years'].corr(df['Monthly_Salary'])
ax.annotate(f'Pearson r = {r:.3f}', xy=(0.98, 0.06), xycoords='axes fraction',
            ha='right', fontsize=10, color=PRIMARY,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=GRID_CLR))

ax.set_title('Experience vs Monthly Salary — Coloured by Education Level', pad=14)
ax.set_xlabel('Years of Experience')
ax.set_ylabel('Monthly Salary (₹)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_inr))
ax.legend(frameon=True, framealpha=0.9, loc='upper left')
ax.grid(True, axis='y', ls='--')
plt.tight_layout()
plt.savefig(CHARTS / 'chart_01_experience_vs_salary.png')
plt.close()
print("✓ Chart 1 saved — Experience vs Salary")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Salary Distribution by Department (box plot)
# ═══════════════════════════════════════════════════════════════════════════════
dept_order = (df.groupby('Department')['Monthly_Salary']
                .median().sort_values(ascending=True).index.tolist())

dept_data = [df.loc[df['Department'] == d, 'Monthly_Salary'].values for d in dept_order]
dept_colours = plt.cm.Blues(np.linspace(0.35, 0.85, len(dept_order)))

fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(dept_data, vert=False, patch_artist=True,
                medianprops=dict(color=ACCENT, lw=2.5),
                whiskerprops=dict(color=PRIMARY, lw=1.2),
                capprops=dict(color=PRIMARY, lw=1.5),
                flierprops=dict(marker='o', color=ACCENT, ms=6, alpha=0.7))
for patch, c in zip(bp['boxes'], dept_colours):
    patch.set_facecolor(c); patch.set_alpha(0.88)

ax.set_yticks(range(1, len(dept_order) + 1))
ax.set_yticklabels(dept_order, fontsize=11)
ax.set_title('Monthly Salary Distribution by Department', pad=14)
ax.set_xlabel('Monthly Salary (₹)')
ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_inr))
ax.grid(True, axis='x', ls='--')
plt.tight_layout()
plt.savefig(CHARTS / 'chart_02_salary_by_department.png')
plt.close()
print("✓ Chart 2 saved — Salary by Department")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Average Salary by Education Level (grouped bar)
# ═══════════════════════════════════════════════════════════════════════════════
edu_stats = (df.groupby('Education_Level')['Monthly_Salary']
               .agg(['mean', 'median']).reindex(edu_order))

fig, ax = plt.subplots(figsize=(10, 6))
x, w = np.arange(len(edu_order)), 0.38
bars_m = ax.bar(x - w/2, edu_stats['mean'],   width=w, color=PRIMARY, alpha=0.90, label='Mean Salary')
bars_d = ax.bar(x + w/2, edu_stats['median'], width=w, color=ACCENT,  alpha=0.90, label='Median Salary')

for bar in bars_m:
    h = bar.get_height()
    ax.annotate(f'₹{h/1000:.0f}K', xy=(bar.get_x()+bar.get_width()/2, h),
                xytext=(0,5), textcoords='offset points', ha='center',
                fontsize=9, color=PRIMARY, fontweight='bold')
for bar in bars_d:
    h = bar.get_height()
    ax.annotate(f'₹{h/1000:.0f}K', xy=(bar.get_x()+bar.get_width()/2, h),
                xytext=(0,5), textcoords='offset points', ha='center',
                fontsize=9, color='#7D3C00', fontweight='bold')

ax.set_xticks(x); ax.set_xticklabels(edu_order, fontsize=11)
ax.set_title('Impact of Education Level on Monthly Salary', pad=14)
ax.set_ylabel('Monthly Salary (₹)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_inr))
ax.legend(frameon=True, framealpha=0.9)
ax.grid(True, axis='y', ls='--', zorder=0)
plt.tight_layout()
plt.savefig(CHARTS / 'chart_03_salary_by_education.png')
plt.close()
print("✓ Chart 3 saved — Salary by Education")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Age vs Monthly Salary (scatter, coloured by Experience)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
sc = ax.scatter(df['Age'], df['Monthly_Salary'],
                c=df['Experience_Years'], cmap='YlOrRd',
                s=90, alpha=0.80, edgecolors='white', lw=0.4, zorder=3)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Years of Experience', color=PRIMARY, fontsize=10)

z = np.polyfit(df['Age'], df['Monthly_Salary'], 2)
xs = np.linspace(df['Age'].min(), df['Age'].max(), 200)
ax.plot(xs, np.poly1d(z)(xs), color=PRIMARY, lw=2.5, ls='--', label='Polynomial Trend')

r = df['Age'].corr(df['Monthly_Salary'])
ax.annotate(f'Pearson r = {r:.3f}', xy=(0.98, 0.06), xycoords='axes fraction',
            ha='right', fontsize=10, color=PRIMARY,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=GRID_CLR))

ax.set_title('Age vs Monthly Salary — Coloured by Years of Experience', pad=14)
ax.set_xlabel('Age (years)'); ax.set_ylabel('Monthly Salary (₹)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_inr))
ax.legend(frameon=True, framealpha=0.9)
ax.grid(True, ls='--', zorder=0)
plt.tight_layout()
plt.savefig(CHARTS / 'chart_04_age_vs_salary.png')
plt.close()
print("✓ Chart 4 saved — Age vs Salary")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Highest & Lowest Paying: Department + City comparison
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# By Department
dept_avg = df.groupby('Department')['Monthly_Salary'].mean().sort_values(ascending=True)
clrs_d   = [GREEN if v == dept_avg.max() else RED if v == dept_avg.min() else PRIMARY
             for v in dept_avg.values]
axes[0].barh(dept_avg.index, dept_avg.values, color=clrs_d, alpha=0.88, edgecolor='white')
for i, (dept, val) in enumerate(dept_avg.items()):
    axes[0].annotate(f'₹{val/1000:.0f}K', xy=(val+500, i),
                     xytext=(4,0), textcoords='offset points', va='center', fontsize=9.5, fontweight='bold')
axes[0].axvline(df['Monthly_Salary'].mean(), color=ACCENT, lw=2, ls='--',
                label=f'Overall Avg: ₹{df["Monthly_Salary"].mean()/1000:.0f}K')
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(fmt_inr))
axes[0].set_title('Avg Salary by Department', pad=12)
axes[0].legend(fontsize=9, framealpha=0.9)
axes[0].grid(True, axis='x', ls='--', zorder=0)

# By City
city_avg = df.groupby('City')['Monthly_Salary'].mean().sort_values(ascending=True)
clrs_c   = [GREEN if v == city_avg.max() else RED if v == city_avg.min() else '#5DADE2'
             for v in city_avg.values]
axes[1].barh(city_avg.index, city_avg.values, color=clrs_c, alpha=0.88, edgecolor='white')
for i, (city, val) in enumerate(city_avg.items()):
    axes[1].annotate(f'₹{val/1000:.0f}K', xy=(val+500, i),
                     xytext=(4,0), textcoords='offset points', va='center', fontsize=9.5, fontweight='bold')
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(fmt_inr))
axes[1].set_title('Avg Salary by City', pad=12)
axes[1].grid(True, axis='x', ls='--', zorder=0)

top_dept = dept_avg.idxmax(); bot_dept = dept_avg.idxmin()
top_city = city_avg.idxmax(); bot_city = city_avg.idxmin()

plt.suptitle('Highest & Lowest Paying: Department and City Comparison',
             fontsize=14, fontweight='bold', color=PRIMARY, y=1.02)
plt.tight_layout()
plt.savefig(CHARTS / 'chart_05_top_bottom_positions.png')
plt.close()
print("✓ Chart 5 saved — Top/Bottom Positions")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Monthly Salary Distribution (histogram + KDE)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(df['Monthly_Salary'], bins=15, color=PRIMARY, alpha=0.72,
        edgecolor='white', lw=0.6, density=True, label='Histogram', zorder=3)

kde = gaussian_kde(df['Monthly_Salary'])
xs  = np.linspace(df['Monthly_Salary'].min(), df['Monthly_Salary'].max(), 300)
ax.plot(xs, kde(xs), color=ACCENT, lw=2.5, label='KDE Curve')
ax.axvline(df['Monthly_Salary'].mean(),   color=GREEN,     lw=2, ls='--',
           label=f'Mean: ₹{df["Monthly_Salary"].mean()/1000:.0f}K')
ax.axvline(df['Monthly_Salary'].median(), color='#9B59B6', lw=2, ls=':',
           label=f'Median: ₹{df["Monthly_Salary"].median()/1000:.0f}K')

ax.set_title('Employee Monthly Salary Distribution', pad=14)
ax.set_xlabel('Monthly Salary (₹)'); ax.set_ylabel('Density')
ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_inr))
ax.legend(frameon=True, framealpha=0.9)
ax.grid(True, axis='y', ls='--', zorder=0)
plt.tight_layout()
plt.savefig(CHARTS / 'chart_06_salary_distribution.png')
plt.close()
print("✓ Chart 6 saved — Salary Distribution")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 7 — Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
num_cols = ['Age', 'Experience_Years', 'education_encoded', 'gender_encoded',
            'department_encoded', 'city_encoded', 'Monthly_Salary']
corr = df[num_cols].corr()
labels = ['Age', 'Experience', 'Education', 'Gender', 'Department', 'City', 'Salary']

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            vmin=-1, vmax=1, ax=ax, xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor='white',
            cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
ax.set_title('Feature Correlation Heatmap', pad=14)
plt.xticks(rotation=30, fontsize=9); plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(CHARTS / 'chart_07_correlation_heatmap.png')
plt.close()
print("✓ Chart 7 saved — Correlation Heatmap")

# ── Print EDA Summary ─────────────────────────────────────────────────────────
print("\n" + "═"*55)
print("  EDA FINDINGS SUMMARY")
print("═"*55)
print(f"  Experience–Salary correlation : r = {df['Experience_Years'].corr(df['Monthly_Salary']):.3f}")
print(f"  Age–Salary correlation        : r = {df['Age'].corr(df['Monthly_Salary']):.3f}")
print(f"  Highest paying department     : {top_dept} (₹{dept_avg.max():,.0f}/mo)")
print(f"  Lowest paying department      : {bot_dept} (₹{dept_avg.min():,.0f}/mo)")
print(f"  Highest paying city           : {top_city} (₹{city_avg.max():,.0f}/mo)")
print(f"  Lowest paying city            : {bot_city} (₹{city_avg.min():,.0f}/mo)")
for edu in edu_order:
    avg = df.loc[df['Education_Level']==edu, 'Monthly_Salary'].mean()
    print(f"  Avg salary — {edu:<12}: ₹{avg:,.0f}/mo")
print("═"*55)
