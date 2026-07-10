import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

df = pd.read_csv('data/global_ecommerce_sales_cleaned.csv', parse_dates=['Order_Date'])

# ---- Feature Engineering for ML ----
features = ['Customer_Segment','Country','Region','Product_Category','Quantity',
            'Unit_Price','Discount_Percent','Shipping_Cost','Month','Quarter','DayOfWeek']
target = 'Total_Sales'

X = df[features]
y = df[target]

categorical = ['Customer_Segment','Country','Region','Product_Category','DayOfWeek']
numeric = ['Quantity','Unit_Price','Discount_Percent','Shipping_Cost','Month','Quarter']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', StandardScaler(), numeric)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42, max_depth=8),
    'Random Forest Regressor': RandomForestRegressor(random_state=42, n_estimators=200, max_depth=12)
}

results = {}
pipelines = {}
for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    pipelines[name] = pipe
    print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")

results_df = pd.DataFrame(results).T.sort_values('R2', ascending=False)
print("\nModel comparison:\n", results_df)

best_model_name = results_df.index[0]
best_pipe = pipelines[best_model_name]
print(f"\nBest model: {best_model_name}")

# Save model comparison chart
fig, ax = plt.subplots(figsize=(8,5))
results_df['R2'].sort_values().plot(kind='barh', color='#2a9d8f', ax=ax)
ax.set_title('Model Comparison — R² Score', fontsize=14, fontweight='bold')
ax.set_xlabel('R² Score')
plt.tight_layout()
plt.savefig('images/09_model_comparison.png')
plt.close()

# Actual vs predicted for best model
best_preds = best_pipe.predict(X_test)
fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(y_test, best_preds, alpha=0.4, color='#264653')
lims = [0, max(y_test.max(), best_preds.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5)
ax.set_xlabel('Actual Sales ($)'); ax.set_ylabel('Predicted Sales ($)')
ax.set_title(f'Actual vs Predicted — {best_model_name}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/10_actual_vs_predicted.png')
plt.close()

# Save best model + metadata
with open('sales_prediction_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_pipe,
        'features': features,
        'categorical': categorical,
        'numeric': numeric,
        'model_name': best_model_name,
        'metrics': results
    }, f)

results_df.to_csv('data/model_comparison_results.csv')
print("\nModel saved to sales_prediction_model.pkl")
