import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("model.pkl")

# Title
st.title("Road Accident Severity Prediction")

# Sidebar inputs
st.sidebar.header("Enter Accident Details")

# Input fields
hour = st.sidebar.slider("Hour of Accident (0-23)", 0, 23, 12)
day_of_week = st.sidebar.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
road_surface = st.sidebar.selectbox("Road Surface Conditions", ["Dry", "Wet", "Snow", "Ice"])
visibility = st.sidebar.selectbox("Visibility", ["Clear", "Fog", "Rain", "Snow"])
weather = st.sidebar.selectbox("Weather Conditions", ["Clear", "Cloudy", "Rain", "Snow", "Fog"])
vehicle_type = st.sidebar.selectbox("Vehicle Type", ["Car", "Motorcycle", "Truck", "Bus", "Bicycle"])
light_conditions = st.sidebar.selectbox("Light Conditions", ["Daylight", "Darkness", "Dusk", "Dawn"])

# Prepare input DataFrame
input_data = {
    "Hour": [hour],
    "Day_of_Week": [day_of_week],
    "Road_Surface": [road_surface],
    "Visibility": [visibility],
    "Weather": [weather],
    "Vehicle_Type": [vehicle_type],
    "Light_Conditions": [light_conditions]
}

input_df = pd.DataFrame(input_data)

# Encoding categorical variables
encoder = LabelEncoder()
for col in input_df.columns:
    if input_df[col].dtype == 'object':
        input_df[col] = encoder.fit_transform(input_df[col])

# Prediction button
if st.button("Predict Severity"):
    try:
        prediction = model.predict(input_df)
        severity_map = {0: "Slight", 1: "Serious", 2: "Fatal"}
        result = severity_map.get(prediction[0], "Unknown")
        st.success(f"Predicted Accident Severity: **{result}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
