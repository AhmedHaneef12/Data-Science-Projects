"""
Task 04 - Part 4: Predictive Modeling & Insights
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, roc_curve, classification_report)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110
np.random.seed(42)

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / 'data'
CHARTS = BASE / 'charts'
REPORTS_DIR = BASE / 'reports'
CHARTS.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / 'cardio_train_features.csv')

# ---------------------------------------------------------
# 1. Prepare features
# ---------------------------------------------------------
feature_cols = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']
X = df[feature_cols].copy()
y = df['cardio'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# ---------------------------------------------------------
# 2. Logistic Regression
# ---------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'precision': precision_score(y_test, y_pred_lr),
    'recall': recall_score(y_test, y_pred_lr),
    'f1': f1_score(y_test, y_pred_lr),
    'roc_auc': roc_auc_score(y_test, y_proba_lr),
}

# ---------------------------------------------------------
# 3. Random Forest
# ---------------------------------------------------------
rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20,
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf),
    'recall': recall_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf),
    'roc_auc': roc_auc_score(y_test, y_proba_rf),
}

results_df = pd.DataFrame(results).T.round(4)
print("MODEL COMPARISON")
print(results_df)

# Pick best model by ROC-AUC
best_model_name = results_df['roc_auc'].idxmax()
print(f"\nBest model: {best_model_name}")

# ---------------------------------------------------------
# 4. Confusion matrices
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, pred, name in zip(axes, [y_pred_lr, y_pred_rf], ['Logistic Regression', 'Random Forest']):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
    ax.set_title(f'{name}\nConfusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(f'{CHARTS}/10_confusion_matrices.png')
plt.close()

# ---------------------------------------------------------
# 5. ROC Curves
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5.5))
for proba, name, color in zip([y_proba_lr, y_proba_rf],
                                ['Logistic Regression', 'Random Forest'],
                                ['#3498db', '#e67e22']):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=color, linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random guess')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve Comparison', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{CHARTS}/11_roc_curve.png')
plt.close()

# ---------------------------------------------------------
# 6. Feature Importance (Random Forest)
# ---------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 6))
importances.plot(kind='barh', color='#2c3e50', ax=ax)
ax.set_title('Feature Importance (Random Forest)', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(f'{CHARTS}/12_feature_importance.png')
plt.close()

top_features = importances.sort_values(ascending=False).head(5)
print("\nTop 5 important features:")
print(top_features)

# ---------------------------------------------------------
# 7. Generate Risk Probability for ALL patients (full dataset)
# using the best model trained above (Random Forest)
# ---------------------------------------------------------
full_proba = rf.predict_proba(X)[:, 1]
df['risk_probability'] = (full_proba * 100).round(1)

def prob_band(p):
    if p < 30: return 'Low'
    elif p < 60: return 'Medium'
    else: return 'High'
df['predicted_risk_band'] = df['risk_probability'].apply(prob_band)

output_cols = ['id', 'age_years', 'gender', 'bmi', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
               'smoke', 'alco', 'active', 'risk_score', 'risk_group',
               'risk_probability', 'predicted_risk_band', 'cardio']
df[output_cols].to_csv(DATA_DIR / 'cardio_risk_predictions.csv', index=False)

band_counts = df['predicted_risk_band'].value_counts()
print(f"\nPredicted risk bands across all {len(df)} patients:")
print(band_counts)

# ---------------------------------------------------------
# 8. Save evaluation report
# ---------------------------------------------------------
with open(REPORTS_DIR / 'model_evaluation.txt', 'w') as f:
    f.write("PREDICTIVE MODELING & EVALUATION\n")
    f.write("="*60 + "\n\n")
    f.write(f"Train/Test split: {len(X_train)} train / {len(X_test)} test (80/20, stratified)\n\n")
    f.write("MODEL COMPARISON\n")
    f.write(results_df.to_string())
    f.write(f"\n\nBest performing model (by ROC-AUC): {best_model_name}\n\n")
    f.write("CLASSIFICATION REPORT - Random Forest\n")
    f.write(classification_report(y_test, y_pred_rf, target_names=['No Disease', 'Disease']))
    f.write("\nTOP 5 FEATURE IMPORTANCES (Random Forest)\n")
    f.write(top_features.to_string())
    f.write(f"\n\nRISK PROBABILITY GENERATED for all {len(df)} patients using Random Forest.predict_proba()\n")
    f.write("Predicted risk band distribution:\n")
    f.write(band_counts.to_string())

print("\nModeling complete. Risk predictions saved to cardio_risk_predictions.csv")
