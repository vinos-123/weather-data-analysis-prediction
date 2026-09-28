# Weather Data Analysis and Prediction

A beginner-friendly Python Machine Learning project that analyzes historical weather data and predicts future temperature trends using **Linear Regression**.

## Features

- Historical weather data analysis
- Temperature trend visualization
- Yearly average temperature analysis
- Linear Regression model
- MAE, RMSE and R² evaluation
- Future temperature prediction for 1–365 days
- Flask web interface
- CSV prediction output
- PNG charts
- GitHub-ready project structure

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Flask
- HTML/CSS

## Project Structure

```text
weather-data-analysis-prediction/
│
├── data/
│   └── weather_data.csv
│
├── output/
│   └── .gitkeep
│
├── screenshots/
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── web_app.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## How the project works

1. Load daily historical weather observations from `data/weather_data.csv`.
2. Convert the date column into a datetime value.
3. Analyze temperature statistics.
4. Plot historical and yearly temperature trends.
5. Create a `DayIndex` representing elapsed time.
6. Train a Linear Regression model:
   - Input: DayIndex
   - Target: Temperature
7. Evaluate the model using MAE, RMSE and R².
8. Generate future dates and predict their temperatures.
9. Display results in the Flask web application.

## Run in VS Code / PowerShell

### 1. Open the project

```powershell
cd D:\
cd weather-data-analysis-prediction
```

### 2. Create virtual environment

```powershell
python -m venv venv
```

### 3. Activate it

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, open Command Prompt and use:

```cmd
venv\Scripts\activate
```

### 4. Install packages

```powershell
python -m pip install -r requirements.txt
```

### 5. Run the analysis

```powershell
python app.py
```

This creates charts and a prediction CSV inside `output/`.

### 6. Run the web application

```powershell
python web_app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Model note

This project intentionally uses a simple Linear Regression baseline so that the ML workflow is easy to understand. Weather is affected by many variables and seasonal patterns, so this model is an educational baseline rather than a production weather forecasting system.

The included CSV is a generated sample dataset for demonstration. For a stronger portfolio project, replace it with a verified real-world historical weather dataset while keeping the same columns or updating the preprocessing code.

## Future improvements

- Add Random Forest Regression
- Add XGBoost/Gradient Boosting
- Add lag features and rolling averages
- Add seasonal features such as month/day-of-year
- Compare multiple models
- Add weather-location selection
- Add real-time weather API integration
- Deploy the Flask application

## Author

**Vinos**  
B.Tech CSE Student
