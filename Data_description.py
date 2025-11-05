#Data description: source, type, structure, metadata, etc.
import pandas as pd
import numpy as np

# ---------------------------
# Load & align the three zones
# ---------------------------
z1 = pd.read_csv("Z1_CAJICA_ambient-weather-20250322-20250925.csv", sep=",")
z2 = pd.read_csv("Z2_GIRALDA_ambient-weather-20250322-20250925.csv", sep=";")
z3 = pd.read_csv("Z3_OIKOS_ambient-weather-20250322-20250925.csv", sep=";")

# Since all the data has the same structure we can analyze one of them
def describe_data(df, zone_name):
    print(f"--- Data Description for {zone_name} ---")
    print("Columns:")
    print(df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nStatistical Summary:")
    print(df.describe())
    print("\nDate Range:")
    print(f"From {df['Date'].min()} to {df['Date'].max()}")
    print("\nSample Data:")
    print(df.head())
    print("\n---------------------------------------\n")

describe_data(z1, "Z1_CAJICA")
describe_data(z2, "Z2_GIRALDA")
describe_data(z3, "Z3_OIKOS")