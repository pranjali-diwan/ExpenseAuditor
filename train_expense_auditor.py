import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/expenses_sample.csv")

# 🔹 Normalize column names
df.columns = df.columns.str.lower().str.strip()

# Separate features & target (update target column if different)
X = df.drop("label", axis=1)   # replace "label" with your actual target col
y = df["label"]

# 🔹 Encode categorical columns
encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# 🔹 Scale numerical column
scaler = StandardScaler()
if "amount" in X.columns:
    X[["amount"]] = scaler.fit_transform(X[["amount"]])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model + preprocessing assets
joblib.dump(model, "models/expense_auditor.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("✅ Model, scaler, and encoders saved successfully in /models/")
