import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "data" / "raw" / "telco_churn_raw.csv"

rng = np.random.default_rng(42)
N = 3000

customer_id = [f"CUST-{10000+i}" for i in range(N)]
gender = rng.choice(["Male", "Female"], N)
senior_citizen = rng.choice([0, 1], N, p=[0.84, 0.16])
partner = rng.choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents = rng.choice(["Yes", "No"], N, p=[0.30, 0.70])

tenure = rng.integers(0, 73, N)

phone_service = rng.choice(["Yes", "No"], N, p=[0.90, 0.10])
multiple_lines = np.where(
    phone_service == "No", "No phone service",
    rng.choice(["Yes", "No"], N, p=[0.42, 0.58])
)

internet_service = rng.choice(["DSL", "Fiber optic", "No"], N, p=[0.35, 0.44, 0.21])

def dep_internet_feature(p_yes):
    out = np.empty(N, dtype=object)
    for i in range(N):
        if internet_service[i] == "No":
            out[i] = "No internet service"
        else:
            out[i] = rng.choice(["Yes", "No"], p=[p_yes, 1 - p_yes])
    return out

online_security = dep_internet_feature(0.35)
online_backup = dep_internet_feature(0.40)
device_protection = dep_internet_feature(0.40)
tech_support = dep_internet_feature(0.35)
streaming_tv = dep_internet_feature(0.45)
streaming_movies = dep_internet_feature(0.45)

contract = rng.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.24, 0.21])
paperless_billing = rng.choice(["Yes", "No"], N, p=[0.59, 0.41])
payment_method = rng.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    N, p=[0.34, 0.23, 0.22, 0.21]
)

base = 18.0
monthly_charges = np.full(N, base)
monthly_charges += (internet_service == "DSL") * rng.normal(25, 3, N)
monthly_charges += (internet_service == "Fiber optic") * rng.normal(45, 5, N)
monthly_charges += (phone_service == "Yes") * rng.normal(8, 2, N)
for col in [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]:
    monthly_charges += (col == "Yes") * rng.normal(5, 1, N)
monthly_charges = np.clip(monthly_charges + rng.normal(0, 3, N), 18, 120).round(2)

total_charges = np.clip(monthly_charges * tenure + rng.normal(0, 20, N), 0, None).round(2)
total_charges = np.where(tenure == 0, 0.0, total_charges)

logit = np.full(N, -1.1)
logit += (contract == "Month-to-month") * 1.4
logit += (contract == "One year") * 0.1
logit += (contract == "Two year") * -1.3
logit += -0.045 * tenure
logit += 0.018 * (monthly_charges - 60)
logit += (internet_service == "Fiber optic") * 0.55
logit += (internet_service == "No") * -0.35
logit += (tech_support == "No") * 0.35
logit += (online_security == "No") * 0.30
logit += (paperless_billing == "Yes") * 0.25
logit += (payment_method == "Electronic check") * 0.35
logit += (senior_citizen == 1) * 0.25
logit += (partner == "No") * 0.15
logit += (dependents == "No") * 0.10
logit += rng.normal(0, 0.6, N)

prob_churn = 1 / (1 + np.exp(-logit))
churn = (rng.random(N) < prob_churn).astype(int)
churn_label = np.where(churn == 1, "Yes", "No")

df = pd.DataFrame({
    "customerID": customer_id,
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn_label,
})

dup_idx = rng.choice(N, 25, replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

missing_idx = rng.choice(df.index, 40, replace=False)
df.loc[missing_idx, "TotalCharges"] = np.nan

missing_idx2 = rng.choice(df.index, 15, replace=False)
df.loc[missing_idx2, "MultipleLines"] = np.nan

df["TotalCharges"] = df["TotalCharges"].apply(lambda x: "" if pd.isna(x) else x)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print("Rows:", len(df))
print(df["Churn"].value_counts(normalize=True))
print("Saved raw dataset to", OUT_PATH)
