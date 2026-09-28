import os

import numpy as np
import pandas as pd

from flask import Flask, render_template, request

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


app = Flask(__name__)


# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

DATA_PATH = os.path.join(
    "data",
    "weather_data.csv"
)

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    "Date"
).reset_index(drop=True)


# ---------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfYear"] = df["Date"].dt.dayofyear

df["Sin_Day"] = np.sin(
    2 * np.pi * df["DayOfYear"] / 365.25
)

df["Cos_Day"] = np.cos(
    2 * np.pi * df["DayOfYear"] / 365.25
)

df["Year_Index"] = (
    df["Year"] - df["Year"].min()
)


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


# ---------------------------------------------------------
# Train Model
# ---------------------------------------------------------

X = df[features]

y = df["Temperature"]

split_index = int(
    len(df) * 0.80
)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

test_predictions = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    test_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

r2 = r2_score(
    y_test,
    test_predictions
)

print("\n=== Model Evaluation ===")
print(f"MAE  : {mae:.2f} °C")
print(f"RMSE : {rmse:.2f} °C")
print(f"R²   : {r2:.3f}")


# ---------------------------------------------------------
# Historical Statistics
# ---------------------------------------------------------

latest_temperature = df.iloc[-1]["Temperature"]

average_temperature = df["Temperature"].mean()

maximum_temperature = df["Temperature"].max()

minimum_temperature = df["Temperature"].min()

latest_date = df.iloc[-1]["Date"].strftime(
    "%d %b %Y"
)


# ---------------------------------------------------------
# Route
# ---------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    predictions = []

    days = 30

    if request.method == "POST":

        try:

            days = int(
                request.form.get(
                    "days",
                    30
                )
            )

            days = max(
                1,
                min(days, 365)
            )

        except ValueError:

            days = 30


        # ---------------------------------------------
        # Future Dates
        # ---------------------------------------------

        last_date = df["Date"].max()

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days,
            freq="D"
        )

        future = pd.DataFrame({
            "Date": future_dates
        })


        # ---------------------------------------------
        # Future Features
        # ---------------------------------------------

        future["Year"] = future["Date"].dt.year

        future["Month"] = future["Date"].dt.month

        future["Day"] = future["Date"].dt.day

        future["DayOfYear"] = (
            future["Date"].dt.dayofyear
        )

        future["Sin_Day"] = np.sin(
            2 * np.pi *
            future["DayOfYear"] /
            365.25
        )

        future["Cos_Day"] = np.cos(
            2 * np.pi *
            future["DayOfYear"] /
            365.25
        )

        future["Year_Index"] = (
            future["Year"] -
            df["Year"].min()
        )


        # ---------------------------------------------
        # Seasonal Weather Averages
        # ---------------------------------------------

        seasonal_weather = df.groupby(
            "DayOfYear"
        )[
            ["Humidity", "WindSpeed", "Rainfall"]
        ].mean()


        future["Humidity"] = future[
            "DayOfYear"
        ].map(
            seasonal_weather["Humidity"]
        )

        future["WindSpeed"] = future[
            "DayOfYear"
        ].map(
            seasonal_weather["WindSpeed"]
        )

        future["Rainfall"] = future[
            "DayOfYear"
        ].map(
            seasonal_weather["Rainfall"]
        )


        # Fill missing values
        future["Humidity"] = future[
            "Humidity"
        ].fillna(
            df["Humidity"].mean()
        )

        future["WindSpeed"] = future[
            "WindSpeed"
        ].fillna(
            df["WindSpeed"].mean()
        )

        future["Rainfall"] = future[
            "Rainfall"
        ].fillna(
            df["Rainfall"].mean()
        )


        # ---------------------------------------------
        # Predict
        # ---------------------------------------------

        future_predictions = model.predict(
            future[features]
        )


        for date, temperature in zip(
            future["Date"],
            future_predictions
        ):

            predictions.append({
                "date": date.strftime(
                    "%d %b %Y"
                ),
                "temperature": round(
                    float(temperature),
                    2
                )
            })


    return render_template(
        "index.html",
        latest_temperature=round(
            latest_temperature,
            2
        ),
        average_temperature=round(
            average_temperature,
            2
        ),
        maximum_temperature=round(
            maximum_temperature,
            2
        ),
        minimum_temperature=round(
            minimum_temperature,
            2
        ),
        latest_date=latest_date,
        predictions=predictions,
        days=days,
        mae=round(mae, 2),
        rmse=round(rmse, 2),
        r2=round(r2, 3)
    )


# ---------------------------------------------------------
# Run Flask
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )