import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------
# Load & align the three zones
# ---------------------------
z1 = pd.read_csv("Z1_CAJICA_ambient-weather-20250322-20250925.csv", sep=",")
z2 = pd.read_csv("Z2_GIRALDA_ambient-weather-20250322-20250925.csv", sep=";")
z3 = pd.read_csv("Z3_OIKOS_ambient-weather-20250322-20250925.csv", sep=";")

for df in (z1, z2, z3):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

def pick(df, tag):
    return df[["Date","Solar Radiation (W/m^2)","Wind Speed (m/sec)"]].rename(columns={
        "Solar Radiation (W/m^2)": f"Solar_{tag}",
        "Wind Speed (m/sec)":      f"Wind_{tag}"
    })

z1 = pick(z1, "Z1")
z2 = pick(z2, "Z2")
z3 = pick(z3, "Z3")

data = z1.merge(z2, on="Date").merge(z3, on="Date").sort_values("Date").reset_index(drop=True)

# ---------------------------------------------------------
# Interpolated series over the triangle (area-weighted mean)
# (With 3 vertices and unknown interior query point, the
# simplest spatial aggregate for time series is the average.)
# ---------------------------------------------------------
data["Solar_interp"] = data[["Solar_Z1","Solar_Z2","Solar_Z3"]].mean(axis=1)
data["Wind_interp"]  = data[["Wind_Z1","Wind_Z2","Wind_Z3"]].mean(axis=1)

# ---------------------------
# Daytime window for forecasts
# ---------------------------
day = "2025-09-25"
mask_day = (data["Date"].dt.date == pd.to_datetime(day).date()) & \
           (data["Date"].dt.hour >= 7) & (data["Date"].dt.hour <= 17)
D = data.loc[mask_day].set_index("Date")

# Guard: ensure enough points
assert len(D) > 30, "Not enough samples in the daytime window to run train/test split."

# ---------------------------------------------------
# Forecasting helpers (AR and Persistence baselines)
# ---------------------------------------------------
def run_ar(series, lags=3, test_points=24):
    series = series.astype(float).fillna(0.0)
    train, test = series.iloc[:-test_points], series.iloc[-test_points:]
    model = AutoReg(train, lags=lags, old_names=False).fit()
    pred = model.predict(start=len(train), end=len(train)+len(test)-1)
    pred.index = test.index
    return test, pred

def run_persistence(series, test_points=24):
    series = series.astype(float).fillna(0.0)
    test = series.iloc[-test_points:]
    last_val = series.iloc[-test_points-1]
    pred = pd.Series(np.full_like(test.values, fill_value=last_val, dtype=float), index=test.index)
    return test, pred

def metrics(y_true, y_pred):
    err = y_true - y_pred
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(((y_true - y_pred)**2).mean())
    r2   = r2_score(y_true, y_pred)
    me   = err.mean()                        # bias (mean error)
    mape = (np.abs(err) / np.maximum(1e-9, np.abs(y_true))).mean() * 100.0
    sde  = err.std(ddof=1)                   # std dev of errors
    return dict(MAE=mae, RMSE=rmse, R2=r2, Bias=me, MAPE=mape, SDE=sde)

# ---------------------------
# 1) Solar: AR vs Persistence
# ---------------------------
solar_true_ar,  solar_pred_ar  = run_ar(D["Solar_interp"], lags=3, test_points=24)
solar_true_pe,  solar_pred_pe  = run_persistence(D["Solar_interp"], test_points=24)

plt.figure(figsize=(12,4))
plt.plot(solar_true_ar.index, solar_true_ar.values, label="True Solar", linewidth=1.0)
plt.plot(solar_pred_ar.index, solar_pred_ar.values, label="AR(3)", linewidth=1.0)
plt.plot(solar_pred_pe.index, solar_pred_pe.values, "--", label="Persistence", linewidth=1.0)
plt.title(f"Solar Radiation Forecasts — {day} (07:00–17:00)")
plt.xlabel("Time (Local)"); plt.ylabel("Solar Radiation (W/m²)")
plt.legend(); plt.tight_layout(); plt.show()

# ---------------------------
# 2) Wind: AR vs Persistence
# ---------------------------
wind_true_ar,   wind_pred_ar   = run_ar(D["Wind_interp"], lags=3, test_points=24)
wind_true_pe,   wind_pred_pe   = run_persistence(D["Wind_interp"], test_points=24)

plt.figure(figsize=(12,4))
plt.plot(wind_true_ar.index, wind_true_ar.values, label="True Wind", linewidth=1.0)
plt.plot(wind_pred_ar.index, wind_pred_ar.values, label="AR(3)", linewidth=1.0)
plt.plot(wind_pred_pe.index, wind_pred_pe.values, "--", label="Persistence", linewidth=1.0)
plt.title(f"Wind Speed Forecasts — {day} (07:00–17:00)")
plt.xlabel("Time (Local)"); plt.ylabel("Wind Speed (m/s)")
plt.legend(); plt.tight_layout(); plt.show()

# ---------------------------
# Summary table
# ---------------------------
rows = []
rows.append({"Series":"Solar","Model":"AR(3)",        **metrics(solar_true_ar, solar_pred_ar)})
rows.append({"Series":"Solar","Model":"Persistence",  **metrics(solar_true_pe, solar_pred_pe)})
rows.append({"Series":"Wind", "Model":"AR(3)",        **metrics(wind_true_ar,  wind_pred_ar)})
rows.append({"Series":"Wind", "Model":"Persistence",  **metrics(wind_true_pe,  wind_pred_pe)})

summary = pd.DataFrame(rows, columns=["Series","Model","MAE","RMSE","R2","Bias","MAPE","SDE"])
print("\n=== Forecast Performance (07:00–17:00, 2025-09-25) ===")
print(summary.to_string(index=False))
