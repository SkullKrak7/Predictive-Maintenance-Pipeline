import pandas as pd

df = pd.read_csv("predictive_maintenance.csv")

print(df["Rotational speed [rpm]"].min(), df["Rotational speed [rpm]"].max())
print(df["Torque [Nm]"].min(), df["Torque [Nm]"].max())
print(df["Tool wear [min]"].min(), df["Tool wear [min]"].max())