# 🌦️ Weather Data Analysis and Prediction

A Machine Learning-based web application for analyzing historical weather data and predicting future temperature trends.

## 📌 Project Overview

Weather Data Analysis and Prediction is a Flask-based Machine Learning project that analyzes historical weather information such as temperature, humidity, wind speed, and rainfall.

The project uses a **Random Forest Regression** model to learn seasonal and weather-related patterns and predict future temperatures.

## 🎯 Objectives

- Analyze historical weather data
- Identify temperature trends
- Analyze humidity, wind speed, and rainfall
- Visualize yearly temperature patterns
- Train a Machine Learning regression model
- Predict future temperature values
- Provide an interactive web dashboard

## 🚀 Features

- 📊 Historical weather data analysis
- 🌡️ Temperature statistics
- 💧 Humidity analysis
- 🌬️ Wind speed analysis
- 🌧️ Rainfall analysis
- 📈 Yearly temperature trend visualization
- 🤖 Random Forest Regression
- 🔮 Future temperature prediction
- 📉 Actual vs Predicted temperature visualization
- 🌐 Flask web dashboard
- 📁 CSV-based data storage

## 🛠️ Technologies Used

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- Random Forest Regression

### Data Analysis
- Pandas
- NumPy

### Data Visualization
- Matplotlib

### Web Development
- Flask
- HTML
- CSS

### Development Tools
- Visual Studio Code
- Git
- GitHub

## 📂 Project Structure

```text
weather-data-analysis-prediction/
│
├── data/
│   └── weather_data.csv
│
├── output/
│   ├── historical_temperature.png
│   ├── yearly_temperature_trend.png
│   ├── actual_vs_predicted.png
│   ├── temperature_prediction.png
│   └── future_temperature_predictions.csv
│
├── static/
│   ├── style.css
│   └── yearly_trend.png
│
├── templates/
│   └── index.html
│
├── app.py
├── web_app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE