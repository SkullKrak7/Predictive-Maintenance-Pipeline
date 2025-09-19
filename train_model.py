import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# 1. Load and clean data
df = pd.read_csv("predictive_maintenance.csv")
df.drop(columns=["UDI", "Product ID"], inplace=True)
df.dropna(inplace=True)

# 2. One-hot encode 'Type', label encode 'Failure Type'
df = pd.get_dummies(df, columns=["Type"], drop_first=True)
label_encoder = LabelEncoder()
df["Failure Type"] = label_encoder.fit_transform(df["Failure Type"])

# 3. Scale numeric features
scaler = StandardScaler()
scaled_features = [
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
df[scaled_features] = scaler.fit_transform(df[scaled_features])

os.makedirs("outputs", exist_ok=True)

# 4. Correlation heatmap for insight
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/corr_heatmap.png")
plt.close()

# 5. Drop 'Air temperature [K]' (high correlation with process temp)
if "Air temperature [K]" in df.columns:
    df.drop(columns=["Air temperature [K]"], inplace=True)

# 6. Define X and y for RF and subsequent feature selection
X_full = df.drop(columns=["Target"])
y_full = df["Target"].values

# 7. Fit RandomForest for feature importances, save bar plot for audit
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_full, y_full)
feature_importance = pd.Series(
    rf.feature_importances_, index=X_full.columns
).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
feature_importance.iloc[::-1].plot(kind="barh", color="skyblue")
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("outputs/feature_importance_rf.png")
plt.close()

# 8. Drop 'Failure Type' (risk of leakage)
if "Failure Type" in df.columns:
    df.drop(columns=["Failure Type"], inplace=True)

# 9. Drop least-important features for practicality—matches earlier logic
for low_imp in ["Type_L", "Type_M"]:
    if low_imp in df.columns:
        df.drop(columns=[low_imp], inplace=True)

# 10. Update X, y after all drops
X = df.drop(columns=["Target"])
y = df["Target"].values

# 11. Train/test split with column names normalized for downstream use
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train.columns = [
    c.strip().replace("[", "").replace("]", "").replace(" ", "_")
    for c in X_train.columns
]
X_test.columns = [
    c.strip().replace("[", "").replace("]", "").replace(" ", "_")
    for c in X_test.columns
]

# 12. Train XGBoost with balanced class weight, using standard params for CI
xgb_model = XGBClassifier(
    eval_metric="logloss",
    tree_method="hist",
    scale_pos_weight=(len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))),
    random_state=42,
)
xgb_model.fit(X_train, y_train)

# 13. Evaluate
y_pred = xgb_model.predict(X_test)
acc = float(accuracy_score(y_test, y_pred))
report = classification_report(y_test, y_pred, output_dict=True)
print(f"Model Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# 14. Save metrics.json (current pipeline style)
metrics = {"accuracy": acc, "report": report}
with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 15. Save final artefact with scaler and feature order for serving
model_info = {
    "model": xgb_model,
    "scaler": scaler,
    "feature_names": list(X_train.columns),
}
with open("model.pkl", "wb") as f:
    pickle.dump(model_info, f)
