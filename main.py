import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_csv("predictive_maintenance.csv")
print("Initial Data Shape:", df.shape)
print(df.head())
