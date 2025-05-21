import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv("predictive_maintenance.csv")
print("Initial Data Shape:", df.shape)
print(df.head())

# Drop unnecessary columns
df.drop(columns=['UDI', 'Product ID'], inplace=True)

# Handle missing values
df.dropna(inplace=True)

# One-hot encode 'Type'
df = pd.get_dummies(df, columns=['Type'], drop_first=True)

# Label encode 'Failure Type' (for later drop)
label_encoder = LabelEncoder()
df['Failure Type'] = label_encoder.fit_transform(df['Failure Type'])

# Drop unwanted or leakage-prone features BEFORE scaling
df.drop(columns=[
    'Air temperature [K]',    # high correlation
    'Failure Type',           # data leakage
    'Type_L', 'Type_M'        # low importance
], inplace=True)

# Scale the remaining features
scaled_features = [
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]
scaler = StandardScaler()
df[scaled_features] = scaler.fit_transform(df[scaled_features])

# Prepare features and target
X = df.drop(columns=['Target'])
y = df['Target']

# Rename columns to safe format
X.rename(columns=lambda x: x.strip().replace("[", "").replace("]", "").replace(" ", "_"), inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = XGBClassifier(
    eval_metric='logloss',
    tree_method='hist',
    device='cuda',
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
)
xgb_model.fit(X_train, y_train)

# Evaluate
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Save model, scaler, and feature names
model_info = {
    "model": xgb_model,
    "scaler": scaler,
    "features": X_train.columns.to_list()
}
with open("model.pkl", "wb") as file:
    pickle.dump(model_info, file)
