import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
import zipfile
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="road-analysis-app",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Functions for Performance ---
@st.cache_resource
def load_model():
    """Load the trained XGBoost model."""
    return joblib.load('xgb_model.pkl')  # Make sure this file is uploaded to Streamlit

@st.cache_data
def load_and_prep_data():
    """Load, clean, and prepare the raw accident data."""
    zip_path = "archive (4).zip"       # ZIP file uploaded to Streamlit
    extract_path = "dataset"
    csv_filename = "AccidentsBig.csv"
    csv_filepath = os.path.join(extract_path, csv_filename)

    # Extract the ZIP if CSV doesn't exist
    if not os.path.exists(csv_filepath):
        if not os.path.exists(zip_path):
            st.error(f"Cannot find '{zip_path}'. Please upload it to the app folder.")
            st.stop()
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    # Load the CSV
    df = pd.read_csv(csv_filepath, low_memory=False)
    df.dropna(subset=['Accident_Severity', 'latitude', 'longitude'], inplace=True)

    # Map severity numbers to labels
    severity_map = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
    df['Severity Label'] = df['Accident_Severity'].map(severity_map)

    # Convert time and extract hour
    df['Time_dt'] = pd.to_datetime(df['Time'], errors='coerce', format='%H:%M')
    df['Hour'] = df['Time_dt'].dt.hour

    return df

# --- Load Assets ---
model = load_model()
df = load_and_prep_data()

# --- User Interface ---
st.title("Road Accident Severity: Prediction & Analysis 🚦")
st.markdown("This interactive dashboard uses an XGBoost model to predict accident severity and visualize high-risk locations across India.")

# --- Sidebar for User Input ---
st.sidebar.header("Simulate an Accident Scenario")
st.sidebar.markdown("Use the controls below to predict the severity of an accident.")

# Create input widgets
hour = st.sidebar.slider("Hour of Day", 0, 23, 17)
day_of_week = st.sidebar.selectbox("Day of Week", options=range(1, 8), format_func=lambda x: ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'][x-1], index=4)
light_conditions = st.sidebar.selectbox("Light Conditions", options=df['Light_Conditions'].unique(), index=0)
weather_conditions = st.sidebar.selectbox("Weather Conditions", options=df['Weather_Conditions'].unique(), index=0)
road_surface = st.sidebar.selectbox("Road Surface Conditions", options=df['Road_Surface_Conditions'].unique(), index=0)
num_vehicles = st.sidebar.number_input("Number of Vehicles Involved", 1, 10, 2)
num_casualties = st.sidebar.number_input("Number of Casualties", 1, 15, 1)

# --- Prediction Logic ---
if st.button("Predict Severity"):
    input_data = {
        "Latitude": latitude,
        "Longitude": longitude,
        "Temperature(F)": temperature,
        "Humidity(%)": humidity,
        "Pressure(in)": pressure,
        "Visibility(mi)": visibility,
        "Wind_Speed(mph)": wind_speed,
        "Distance(mi)": distance,
        "Side": side,
        "Amenity": amenity,
        "Bump": bump,
        "Crossing": crossing,
        "Give_Way": give_way,
        "Junction": junction,
        "No_Exit": no_exit,
        "Railway": railway,
        "Roundabout": roundabout,
        "Station": station,
        "Stop": stop,
        "Traffic_Calming": traffic_calming,
        "Traffic_Signal": traffic_signal,
        "Turning_Loop": turning_loop,
        "Start_Hour": start_hour,
        "Weekday": weekday,
        "Month": month,
        "Year": year
    }

    input_features = pd.DataFrame([input_data])

    # ✅ Align columns with model's expected features
    expected_cols = list(model.feature_names_in_)  # or your saved expected columns list
    for col in expected_cols:
        if col not in input_features.columns:
            input_features[col] = 0  # Default value for missing columns

    # Reorder columns
    input_features = input_features[expected_cols]

    # ✅ Make prediction
    prediction = model.predict(input_features)[0]
    st.success(f"Predicted Accident Severity: {prediction}")

    )

    prediction_index = model.predict(input_features)[0]
    prediction_proba = model.predict_proba(input_features)[0]
    severity_labels = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}
    predicted_severity = severity_labels[prediction_index]

    st.subheader("Prediction Result")
    if predicted_severity == 'Fatal':
        st.error(f"Predicted Severity: *{predicted_severity}* (Probability: {prediction_proba[prediction_index]:.2%})")
    elif predicted_severity == 'Serious':
        st.warning(f"Predicted Severity: *{predicted_severity}* (Probability: {prediction_proba[prediction_index]:.2%})")
    else:
        st.success(f"Predicted Severity: *{predicted_severity}* (Probability: {prediction_proba[prediction_index]:.2%})")

# --- Interactive Map Visualization ---
st.subheader("Interactive Map of Accident Hotspots")
selected_severity_map = st.selectbox("Select Severity to Visualize:", options=df['Severity Label'].unique())
map_data = df[df['Severity Label'] == selected_severity_map]

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/dark-v9',
    initial_view_state=pdk.ViewState(
        latitude=20.5937,
        longitude=78.9629,
        zoom=4,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
            'HeatmapLayer',
            data=map_data,
            get_position='[longitude, latitude]',
            radius=100,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
        ),
    ],
))
