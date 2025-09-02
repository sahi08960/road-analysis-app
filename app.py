import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# App title and header
st.set_page_config(page_title="Road Accident Severity Predictor", layout="centered")

# Add custom CSS for colorful design
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
        padding: 20px;
        border-radius: 10px;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        font-size: 18px;
        padding: 10px 24px;
        border: none;
        cursor: pointer;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚦 Road Accident Severity Predictor")

st.write("### Enter Accident Details:")

# Input fields
hour = st.slider("Hour of the Day (0-23)", 0, 23, 12)
vehicle_type = st.selectbox("Vehicle Type", ["Car", "Bike", "Truck", "Bus"])
road_type = st.selectbox("Road Type", ["Highway", "City Road", "Rural Road"])
weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Foggy", "Snowy"])
lighting = st.selectbox("Lighting Condition", ["Daylight", "Darkness", "Street Lights"])
speed_limit = st.number_input("Speed Limit (km/h)", 20, 120, 60)

# Prepare input for model
input_data = pd.DataFrame({
    'Hour': [hour],
    'Vehicle_Type': [vehicle_type],
    'Road_Type': [road_type],
    'Weather': [weather],
    'Lighting': [lighting],
    'Speed_Limit': [speed_limit]
})

# Predict button
if st.button("🔮 Predict Severity"):
    prediction = model.predict(input_data)[0]
    
    # Display result with color
    if prediction == 0:
        st.success("✅ Low Severity Accident")
    elif prediction == 1:
        st.warning("⚠️ Medium Severity Accident")
    else:
        st.error("🔥 High Severity Accident")
