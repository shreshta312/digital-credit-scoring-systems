import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/rural_credit_dataset_mixed.csv")

# Remove ID column if present
if "borrower_id" in df.columns:
    df = df.drop(columns=["borrower_id"])

# Target column
target = "loan_repayment_status"

X = df.drop(columns=[target])
y = df[target]

# Categorical columns
categorical_cols = [
    "gender",
    "education_level",
    "village_state",
    "occupation_type",
    "irrigation_type",
    "crop_type"
]

# Numerical columns
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Column transformer
ct = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# Transform input data
X_transformed = ct.fit_transform(X)

# Scale transformed data
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X_transformed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train XGBoost model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Save model files
joblib.dump(model, "models/xgb_model.pkl")
joblib.dump(ct, "models/column_transformer.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model, column transformer, and scaler saved successfully.")