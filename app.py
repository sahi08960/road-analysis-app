import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load model
model = joblib.load("xgb_model.pkl")

st.title("Bike Demand Prediction")

# ---- User Inputs ----
st.header("Enter Ride Details")

# Date and time input
date = st.date_input("Date")
time = st.time_input("Time")

# Extract features from date and time
hour = time.hour  # Only hour now (Month and Weekday removed)

temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
windspeed = st.number_input("Windspeed (km/h)", min_value=0.0, max_value=100.0, value=10.0)

season = st.selectbox("Season", ["spring", "summer", "fall", "winter"])
weather = st.selectbox("Weather", ["clear", "cloudy", "rainy", "snowy"])

# ---- Create input features ----
input_data = {
    "hour": hour,
    "temperature": temperature,
    "humidity": humidity,
    "windspeed": windspeed,
    "season": season,
    "weather": weather
}

input_features = pd.DataFrame([input_data])

# One-hot encoding for categorical variables
input_features = pd.get_dummies(input_features)

# ---- Align with model features ----
expected_cols = model.get_booster().feature_names  # Get features from model
input_features = input_features.reindex(columns=expected_cols, fill_value=0)

# ---- Predict ----
if st.button("Predict"):
    prediction = model.predict(input_features)
    st.success(f"Predicted Bike Demand: {int(prediction[0])} rides")
