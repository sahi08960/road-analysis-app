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

# ✅ Add Custom CSS for Background & Button Styling
st.markdown("""
    <style>
    /* Main page background */
    .main {
        background-color: #FFD700; /* Yellow */
    }

    /* Headings color */
    h1, h2, h3 {
        color: #ff4b4b;
    }

    /* Predict Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #ff4b4b, #ff9800);
        color: white;
        border-radius: 12px;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 28px;
        border: none;
        cursor: pointer;
        transition: all 0.4s ease-in-out;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ff9800, #ff4b4b);
        color: #fff;
        transform: scale(1.07);
    }

    /* Sidebar starry background */
    section[data-testid="stSidebar"] {
        background: radial-gradient(ellipse at bottom, #1b2735 0%, #090a0f 100%);
        position: relative;
        overflow: hidden;
        color: white;
    }

    /* Stars effect */
    section[data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        width: 2px;
        height: 2px;
        background: white;
        box-shadow:
            50px 80px white,
            100px 150px white,
            150px 250px white,
            200px 100px white,
            250px 180px white,
            300px 200px white,
            350px 120px white;
        animation: starryMove 50s linear infinite;
    }

    @keyframes starryMove {
        from { transform: translateY(0px); }
        to { transform: translateY(-1000px); }
    }
    </style>
""", unsafe_allow_html=True)

# --- Caching Functions ---
@st.cache_resource
def load_model():
    """Load the trained XGBoost or Pipeline model."""
    return joblib.load('xgb_model.pkl')

@st.cache_data
def load_and_prep_data():
    """Load, clean, and prepare the raw accident data."""
    zip_path = "archive (4).zip"
    extract_path = "dataset"
    csv_filename = "AccidentsBig.csv"
    csv_filepath = os.path.join(extract_path, csv_filename)

    # Extract ZIP if CSV doesn't exist
    if not os.path.exists(csv_filepath):
        if not os.path.exists(zip_path):
            st.error(f"Cannot find '{zip_path}'. Please upload it to the app folder.")
            st.stop()
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    # Load CSV
    df = pd.read_csv(csv_filepath, low_memory=False)
    df.dropna(subset=['Accident_Severity', 'latitude', 'longitude'], inplace=True)

    severity_map = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
    df['Severity Label'] = df['Accident_Severity'].map(severity_map)

    # Convert time and extract hour
    df['Time_dt'] = pd.to_datetime(df['Time'], errors='coerce', format='%H:%M')
    df['Hour'] = df['Time_dt'].dt.hour

    return df

# --- Load Model & Data ---
model = load_model()
df = load_and_prep_data()

# --- UI ---
st.title("Road Accident Severity: Prediction & Analysis 🚦")
st.markdown("This dashboard predicts accident severity using an XGBoost model and visualizes accident hotspots.")

# --- Sidebar Inputs ---
st.sidebar.header("Simulate an Accident Scenario")
hour = st.sidebar.slider("Hour of Day", 0, 23, 17)
day_of_week = st.sidebar.selectbox("Day of Week", options=range(1, 8),
                                   format_func=lambda x: ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'][x-1], index=4)
light_conditions = st.sidebar.selectbox("Light Conditions", options=df['Light_Conditions'].dropna().unique(), index=0)
weather_conditions = st.sidebar.selectbox("Weather Conditions", options=df['Weather_Conditions'].dropna().unique(), index=0)
road_surface = st.sidebar.selectbox("Road Surface Conditions", options=df['Road_Surface_Conditions'].dropna().unique(), index=0)
num_vehicles = st.sidebar.number_input("Number of Vehicles Involved", 1, 10, 2)
num_casualties = st.sidebar.number_input("Number of Casualties", 1, 15, 1)

# --- Prediction Logic ---
if st.button("🔮 Predict Severity"):
    try:
        # Create input data
        input_data = {
            "Hour": hour,
            "Day_of_Week": day_of_week,
            "Light_Conditions": light_conditions,
            "Weather_Conditions": weather_conditions,
            "Road_Surface_Conditions": road_surface,
            "Number_of_Vehicles": num_vehicles,
            "Number_of_Casualties": num_casualties
        }

        input_df = pd.DataFrame([input_data])

        # Check if the model is pipeline or raw model
        if hasattr(model, "feature_names_in_"):
            # Align with model features
            input_df_encoded = pd.get_dummies(input_df)

            expected_cols = list(model.feature_names_in_)
            for col in expected_cols:
                if col not in input_df_encoded.columns:
                    input_df_encoded[col] = 0

            input_df_encoded = input_df_encoded[expected_cols]
        else:
            # If model is a pipeline, no encoding needed
            input_df_encoded = input_df

        # Prediction
        prediction_index = model.predict(input_df_encoded)[0]

        # Check if model supports predict_proba
        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(input_df_encoded)[0]
        else:
            prediction_proba = [0, 0, 0]

        severity_labels = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}
        predicted_severity = severity_labels.get(prediction_index, "Unknown")

        # Show result
        st.subheader("Prediction Result")
        if predicted_severity == 'Fatal':
            st.error(f"Predicted Severity: *{predicted_severity}* (Probability: {prediction_proba[prediction_index]:.2%})")
        elif predicted_severity == 'Serious':
            st.warning(f"Predicted Severity: *{predicted_severity}* (Probability: {prediction_proba[prediction_index]:.2%})")
        else:
            st.success(f"Predicted Severity: *{predicted_severity}* (Probability: {prediction_proba[prediction_index]:.2%})")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# --- Map Visualization ---
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

