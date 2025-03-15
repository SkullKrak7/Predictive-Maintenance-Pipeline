import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#load data
df = pd.read_csv("predictive_maintenance.csv")
print("Initial Data Shape:", df.shape)
print(df.head())

#drop unnecessary columns
df.drop(columns=['UDI', 'Product ID'], inplace=True)
print("Data Shape after dropping useless columns:", df.shape)

#handle missing values
print("Missing Values Before Drop:")
print(df.isna().sum())
df.dropna(inplace=True)
print("Missing Values After Drop:")
print(df.isna().sum())

#make a histogram for frequency distribution
df.hist(figsize=(12, 6))
plt.show()

#perform one-hot encoding on type column
df = pd.get_dummies(df, columns=['Type'], drop_first=True)

#perform label encoding on failure type column
label_encoder = LabelEncoder()
df['Failure Type'] = label_encoder.fit_transform(df['Failure Type'])

#scale to normalise the data
scaler = StandardScaler()
scaled_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
df[scaled_features] = scaler.fit_transform(df[scaled_features])

#make a heatmap to see feature correlation
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

#drop air temperature as it highly correlates with process temperature
df.drop(columns=['Air temperature [K]'], inplace=True)

#define features and target
X = df.drop(columns=['Target'])
y = df['Target']

#perform feature importance analysis using random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
feature_importance.plot(kind="bar", color="skyblue")
plt.title("Feature Importance - Random Forest")
plt.show()

#drop failure type due to chances of data leakage
df.drop(columns=['Failure Type'], inplace=True)

#drop less important features
df.drop(columns=['Type_L', 'Type_M'], inplace=True)

#update features and target
X = df.drop(columns=['Target'])
y = df['Target']

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)