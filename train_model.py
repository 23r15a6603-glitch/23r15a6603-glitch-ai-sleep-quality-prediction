import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

# Load data
df = pd.read_csv("ai_sleep_quality_dataset_2000.csv")

# Encode categorical features
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
df["Smoking"] = df["Smoking"].map({"No": 0, "Yes": 1})
df["Sleep Disorder History"] = df["Sleep Disorder History"].map({"No": 0, "Yes": 1})
df["Wake-up Consistency"] = df["Wake-up Consistency"].map({"Inconsistent": 0, "Consistent": 1})

# Features and target
X = df.drop("Sleep Quality", axis=1)
y = df["Sleep Quality"].map({"Poor": 0, "Fair": 1, "Good": 2})  # ✅ Encode labels

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "xgb_sleep_quality_model.pkl")
joblib.dump(scaler, "scaler_sleep_quality.pkl")

print("✅ Model and scaler saved successfully.")
