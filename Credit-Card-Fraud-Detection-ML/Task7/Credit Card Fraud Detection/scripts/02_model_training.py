"""
02_model_training.py
---------------------
Data Science Task #07 - Credit Card Fraud Detection
Step 2: Machine Learning Model Development

- Prepares the data (feature scaling)
- Splits into train/test sets (stratified, to preserve the tiny fraud class)
- Handles class imbalance using RANDOM OVERSAMPLING of the minority
  (fraud) class on the TRAINING set only (sklearn.utils.resample),
  combined with class_weight='balanced' where supported.
  NOTE: imbalanced-learn (SMOTE) is not available in this offline
  environment, so manual oversampling via sklearn is used instead -
  this is one of the techniques explicitly allowed by the task brief.
- Trains 4 classification models: Logistic Regression, Decision Tree,
  Random Forest, K-Nearest Neighbors
- Saves trained models + scaler + train/test splits for the evaluation step
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 42
DATA_PATH = "data/creditcard_cleaned.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. Load cleaned data
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Class"])
y = df["Class"]

# ---------------------------------------------------------------
# 2. Feature scaling (Time & Amount are unscaled in the raw PCA data;
#    we scale ALL numeric features for distance-based models like KNN)
# ---------------------------------------------------------------
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
joblib.dump(scaler, f"{MODELS_DIR}/scaler.joblib")

# ---------------------------------------------------------------
# 3. Train / test split (stratified to keep at least 1 fraud case
#    in both train and test despite the extreme rarity of fraud
#    in this 1,000-row sample)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
print("Train class distribution:\n", y_train.value_counts())
print("Test class distribution:\n", y_test.value_counts())

# ---------------------------------------------------------------
# 4. Handle class imbalance: random oversampling of the minority
#    class in the TRAINING data only (never touch the test set)
# ---------------------------------------------------------------
train_df = X_train.copy()
train_df["Class"] = y_train.values

majority = train_df[train_df.Class == 0]
minority = train_df[train_df.Class == 1]

minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=RANDOM_STATE,
)

train_balanced = pd.concat([majority, minority_upsampled]).sample(
    frac=1, random_state=RANDOM_STATE
).reset_index(drop=True)

X_train_bal = train_balanced.drop(columns=["Class"])
y_train_bal = train_balanced["Class"]

print("\nBalanced training class distribution (after oversampling):\n",
      y_train_bal.value_counts())

# Persist splits for the evaluation script / notebook
X_train.to_csv(f"{MODELS_DIR}/X_train.csv", index=False)
X_test.to_csv(f"{MODELS_DIR}/X_test.csv", index=False)
y_train.to_csv(f"{MODELS_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{MODELS_DIR}/y_test.csv", index=False)
X_train_bal.to_csv(f"{MODELS_DIR}/X_train_balanced.csv", index=False)
y_train_bal.to_csv(f"{MODELS_DIR}/y_train_balanced.csv", index=False)

# ---------------------------------------------------------------
# 5. Train models on the oversampled training set
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, max_depth=6
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, max_depth=8
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
}

for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    fname = name.lower().replace(" ", "_").replace("-", "_")
    joblib.dump(model, f"{MODELS_DIR}/{fname}.joblib")
    print(f"Trained and saved: {name} -> {MODELS_DIR}/{fname}.joblib")

print("\nStep 2 (Model Development) complete. All models saved in 'models/'.")
