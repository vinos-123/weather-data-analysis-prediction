import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = os.path.join("data", "weather_data.csv")
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)


print("\n=== Weather Data Summary ===")
print(df.describe(include="all"))


# ---------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfYear"] = df["Date"].dt.dayofyear

# Seasonal features
df["Sin_Day"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.25)
df["Cos_Day"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.25)

# Long-term trend
df["Year_Index"] = df["Year"] - df["Year"].min()


# ---------------------------------------------------------
# Average Temperature by Year
# ---------------------------------------------------------

yearly_temperature = df.groupby("Year")["Temperature"].mean()

print("\n=== Average Temperature by Year ===")
print(yearly_temperature.round(2))


# ---------------------------------------------------------
# Prepare Features and Target
# ---------------------------------------------------------

features = [
    "Year_Index",
    "Month",
    "DayOfYear",
    "Sin_Day",
    "Cos_Day",
    "Humidity",
    "WindSpeed",
    "Rainfall"
]

X = df[features]
y = df["Temperature"]


# ---------------------------------------------------------
# Time-Based Train/Test Split
# ---------------------------------------------------------

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# ---------------------------------------------------------
# Random Forest Regression Model
# ---------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ---------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== Model Evaluation ===")
print(f"MAE  : {mae:.2f} °C")
print(f"RMSE : {rmse:.2f} °C")
print(f"R²   : {r2:.3f}")


# ---------------------------------------------------------
# Historical Temperature Plot
# ---------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["Date"],
    df["Temperature"],
    linewidth=1
)

plt.title("Historical Temperature")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "historical_temperature.png"),
    dpi=150
)

plt.close()


# ---------------------------------------------------------
# Yearly Temperature Trend
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    yearly_temperature.index,
    yearly_temperature.values,
    marker="o"
)

plt.title("Average Temperature by Year")
plt.xlabel("Year")
plt.ylabel("Average Temperature (°C)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "yearly_temperature_trend.png"),
    dpi=150
)

plt.close()


# ---------------------------------------------------------
# Test Predictions Plot
# ---------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["Date"].iloc[split_index:],
    y_test,
    label="Actual Temperature"
)

plt.plot(
    df["Date"].iloc[split_index:],
    y_pred,
    label="Predicted Temperature"
)

plt.title("Actual vs Predicted Temperature")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"),
    dpi=150
)

plt.close()


# ---------------------------------------------------------
# Future Prediction
# ---------------------------------------------------------

future_days = 30

last_date = df["Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=future_days,
    freq="D"
)

future = pd.DataFrame({
    "Date": future_dates
})

future["Year"] = future["Date"].dt.year
future["Month"] = future["Date"].dt.month
future["Day"] = future["Date"].dt.day
future["DayOfYear"] = future["Date"].dt.dayofyear

future["Sin_Day"] = np.sin(
    2 * np.pi * future["DayOfYear"] / 365.25
)

future["Cos_Day"] = np.cos(
    2 * np.pi * future["DayOfYear"] / 365.25
)

future["Year_Index"] = (
    future["Year"] - df["Year"].min()
)


# Use historical seasonal averages for future weather variables
seasonal_weather = df.groupby("DayOfYear")[
    ["Humidity", "WindSpeed", "Rainfall"]
].mean()

future["Humidity"] = future["DayOfYear"].map(
    seasonal_weather["Humidity"]
)

future["WindSpeed"] = future["DayOfYear"].map(
    seasonal_weather["WindSpeed"]
)

future["Rainfall"] = future["DayOfYear"].map(
    seasonal_weather["Rainfall"]
)


# Handle leap-year day if necessary
future["Humidity"] = future["Humidity"].fillna(
    df["Humidity"].mean()
)

future["WindSpeed"] = future["WindSpeed"].fillna(
    df["WindSpeed"].mean()
)

future["Rainfall"] = future["Rainfall"].fillna(
    df["Rainfall"].mean()
)


future_predictions = model.predict(
    future[features]
)

future["PredictedTemperature"] = future_predictions


print("\n=== Next 30 Days Prediction ===")

print(
    future[
        ["Date", "PredictedTemperature"]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# Save Future Predictions
# ---------------------------------------------------------

prediction_file = os.path.join(
    OUTPUT_DIR,
    "future_temperature_predictions.csv"
)

future[
    ["Date", "PredictedTemperature"]
].to_csv(
    prediction_file,
    index=False
)


# ---------------------------------------------------------
# Future Prediction Plot
# ---------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    future["Date"],
    future["PredictedTemperature"],
    marker="o"
)

plt.title("Next 30 Days Temperature Prediction")
plt.xlabel("Date")
plt.ylabel("Predicted Temperature (°C)")
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "temperature_prediction.png"
    ),
    dpi=150
)

plt.close()


print(
    f"\nOutput files saved to: "
    f"{os.path.abspath(OUTPUT_DIR)}"
)