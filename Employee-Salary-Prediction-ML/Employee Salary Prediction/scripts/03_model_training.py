"""
Task #06 - Employee Salary Prediction
Script 03: Machine Learning Model Development & Evaluation
Dataset : Employee Salary Dataset — Kaggle
Source  : https://www.kaggle.com/datasets/prince7489/employee-salary-dataset/data
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE   = Path(__file__).parent.parent
CLEAN  = BASE / 'data' / 'cleaned_employee_salary.csv'
MODELS = BASE / 'models'
CHARTS = BASE / 'charts'

PRIMARY = '#1E3A5F'; ACCENT = '#E07B39'; GREEN = '#27AE60'
LIGHT_BG = '#F7F9FC'; GRID_CLR = '#DDE4ED'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'axes.facecolor': LIGHT_BG, 'figure.facecolor': 'white',
    'grid.color': GRID_CLR, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

print("=" * 65)
print("  TASK #06 — MACHINE LEARNING MODEL DEVELOPMENT")
print("=" * 65)

df = pd.read_csv(CLEAN)
print(f"\n[1] Loaded: {df.shape}")

NUMERIC = ['Age', 'Experience_Years']
CATEG   = ['Department', 'Education_Level', 'Gender', 'City']
TARGET  = 'Monthly_Salary'

X = df[NUMERIC + CATEG]
y = df[TARGET]
print(f"\n[2] Features  → Numeric: {NUMERIC}")
print(f"              → Categorical: {CATEG}")
print(f"              → Target: {TARGET}  (₹{y.min():,} – ₹{y.max():,})")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"\n[3] Train/Test Split  → Train: {len(X_train)} | Test: {len(X_test)}")
print(f"    ⚠  Small dataset (50 records) — CV used alongside hold-out evaluation")

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), NUMERIC),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEG),
])

models = {
    'Linear Regression': Pipeline([('pre', preprocessor), ('m', LinearRegression())]),
    'Decision Tree':     Pipeline([('pre', preprocessor), ('m', DecisionTreeRegressor(
                                   max_depth=4, min_samples_split=4, min_samples_leaf=2, random_state=42))]),
    'Random Forest':     Pipeline([('pre', preprocessor), ('m', RandomForestRegressor(
                                   n_estimators=100, max_depth=4, min_samples_split=4,
                                   min_samples_leaf=2, random_state=42, n_jobs=-1))]),
}

results = []
print(f"\n[4] Training & Evaluation\n" + "-"*65)

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    yp_test  = pipeline.predict(X_test)
    yp_train = pipeline.predict(X_train)

    mae  = mean_absolute_error(y_test, yp_test)
    rmse = np.sqrt(mean_squared_error(y_test, yp_test))
    mse  = mean_squared_error(y_test, yp_test)
    r2   = r2_score(y_test, yp_test)
    r2tr = r2_score(y_train, yp_train)
    cv   = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

    results.append({'Model': name, 'MAE (₹)': round(mae,2), 'MSE (₹²)': round(mse,2),
                    'RMSE (₹)': round(rmse,2), 'R² Test': round(r2,4),
                    'R² Train': round(r2tr,4), 'CV R² Mean': round(cv.mean(),4),
                    'CV R² Std': round(cv.std(),4)})

    print(f"\n  ▶ {name}")
    print(f"    MAE       : ₹{mae:>10,.2f}")
    print(f"    RMSE      : ₹{rmse:>10,.2f}")
    print(f"    R² Test   : {r2:>10.4f}")
    print(f"    R² Train  : {r2tr:>10.4f}")
    print(f"    CV R² 5-fold: {cv.mean():.4f} ± {cv.std():.4f}")

print("-"*65)
results_df = pd.DataFrame(results)

# Best = highest R² on test set
best_idx  = results_df['R² Test'].idxmax()
best_name = results_df.loc[best_idx, 'Model']
best_r2   = results_df.loc[best_idx, 'R² Test']
best_rmse = results_df.loc[best_idx, 'RMSE (₹)']
best_cv   = results_df.loc[best_idx, 'CV R² Mean']

print(f"\n[5] Model Comparison")
print(results_df[['Model','MAE (₹)','RMSE (₹)','R² Test','CV R² Mean']].to_string(index=False))
print(f"\n[6] Best Model (highest R² on test): {best_name}")
print(f"    R² Test : {best_r2:.4f}  |  RMSE: ₹{best_rmse:,.2f}")
print(f"    ⚠  Low R² reflects small sample size (50 rows).")
print(f"       The pipeline is fully functional; larger data will improve accuracy.")

# ── Feature Importance (Random Forest) ───────────────────────────────────────
rf_pipe = models['Random Forest']
ohe     = rf_pipe.named_steps['pre'].named_transformers_['cat']   # direct OHE
ohe_names = ohe.get_feature_names_out(CATEG)
all_feat  = NUMERIC + list(ohe_names)
importances = rf_pipe.named_steps['m'].feature_importances_

imp_dict = {}
for fname, imp in zip(all_feat, importances):
    root = next((c for c in CATEG if fname.startswith(c)), fname if fname in NUMERIC else None)
    if root:
        imp_dict[root] = imp_dict.get(root, 0) + imp
fi = pd.Series(imp_dict).sort_values(ascending=False)
print(f"\n[7] RF Feature Importances")
for feat, imp in fi.items():
    print(f"    {feat:<22} {imp:.4f}  {'█'*int(imp*60)}")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(models[best_name], MODELS / 'best_model.pkl')
joblib.dump({
    'model_name': best_name, 'numeric': NUMERIC, 'categorical': CATEG,
    'target': TARGET, 'r2': best_r2, 'cv_r2': best_cv, 'rmse': best_rmse,
    'departments': ['Finance','HR','IT','Marketing','Operations'],
    'cities': ['Bangalore','Chennai','Delhi','Hyderabad','Mumbai'],
    'education_levels': ['High School','Bachelor','Master','PhD'],
    'mean_salary': float(y.mean()),
}, MODELS / 'model_metadata.pkl')
results_df.to_csv(MODELS / 'model_comparison.csv', index=False)
print(f"\n✓ Saved best model, metadata, comparison CSV")

# ── Chart 8: Model Comparison ────────────────────────────────────────────────
names = results_df['Model'].tolist()
r2s   = results_df['R² Test'].tolist()
rmses = results_df['RMSE (₹)'].tolist()
cvs   = results_df['CV R² Mean'].tolist()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, vals, title, lower_better in [
    (axes[0], r2s,   'R² Score (Test Set)',    False),
    (axes[1], rmses, 'RMSE in ₹ (lower=better)', True),
    (axes[2], cvs,   'CV R² Mean (5-fold)',    False),
]:
    best_val = min(vals) if lower_better else max(vals)
    clrs = [GREEN if v == best_val else PRIMARY for v in vals]
    bars = ax.bar(names, vals, color=clrs, alpha=0.88, edgecolor='white', width=0.52)
    for bar, v in zip(bars, vals):
        lbl = f'₹{v/1000:.1f}K' if lower_better else f'{v:.3f}'
        ax.annotate(lbl, xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0,5), textcoords='offset points', ha='center',
                    fontsize=9.5, fontweight='bold', color=GREEN if v==best_val else PRIMARY)
    if lower_better:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x/1000:.0f}K'))
    ax.set_title(title, pad=10)
    ax.tick_params(axis='x', rotation=12)
    ax.grid(True, axis='y', ls='--', zorder=0)
    ax.axhline(0, color='gray', lw=0.8, ls='-')

plt.suptitle('Model Performance Comparison — Kaggle Employee Salary Dataset',
             fontsize=13, fontweight='bold', color=PRIMARY, y=1.02)
plt.tight_layout()
plt.savefig(CHARTS / 'chart_08_model_comparison.png')
plt.close()
print("✓ Chart 8 saved")

# ── Chart 9: Feature Importance ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
clrs = [ACCENT if v == fi.max() else PRIMARY for v in fi.values[::-1]]
ax.barh(fi.index[::-1], fi.values[::-1], color=clrs[::-1], alpha=0.88, edgecolor='white', zorder=3)
for i, (feat, val) in enumerate(zip(fi.index[::-1], fi.values[::-1])):
    ax.annotate(f'{val:.3f}', xy=(val, i), xytext=(4,0),
                textcoords='offset points', va='center', fontsize=10, fontweight='bold')
ax.set_title('Random Forest — Feature Importance for Salary Prediction', pad=14)
ax.set_xlabel('Importance Score')
ax.grid(True, axis='x', ls='--', zorder=0)
plt.tight_layout()
plt.savefig(CHARTS / 'chart_09_feature_importance.png')
plt.close()
print("✓ Chart 9 saved")
print(f"\n{'='*65}\n  COMPLETE — Best: {best_name}  R²={best_r2:.4f}  RMSE=₹{best_rmse:,.0f}\n{'='*65}")
