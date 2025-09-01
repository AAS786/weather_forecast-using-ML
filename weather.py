import streamlit as st
import pickle as pk
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Load the trained model
model = pk.load(open('weather.sav', 'rb'))

# Title
st.title("🌦️ Weather Forecast Prediction App")

st.write("Enter the weather details below to get prediction:")

# Create columns for inputs
col1, col2, col3 = st.columns(3)

with col1:
    Temperature = st.number_input("🌡️ Temperature (°C)", min_value=-50.0, max_value=60.0, step=0.1)
    Wind_Speed = st.number_input("💨 Wind Speed (km/h)", min_value=0.0, max_value=300.0, step=0.1)
    Cloud_Cover = st.selectbox("☁️ Cloud Cover", ["Clear", "Partly Cloudy", "Overcast"])

with col2:
    Humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
    Precipitation = st.number_input("🌧️ Precipitation (%)", min_value=0.0, max_value=100.0, step=0.1)
    Season = st.selectbox("🍂 Season", ["Winter", "Spring", "Summer", "Autumn"])

with col3:
    Atmospheric_Pressure = st.number_input("🌬️ Atmospheric Pressure (hPa)", min_value=800.0, max_value=1100.0, step=0.1)
    UV_Index = st.number_input("🔆 UV Index", min_value=0.0, max_value=15.0, step=0.1)
    Visibility = st.number_input("👀 Visibility (km)", min_value=0.0, max_value=50.0, step=0.1)
    Location = st.selectbox("📍 Location", ["Urban", "Rural", "Coastal", "Mountain"])

# Prediction button
if st.button("🚀 Predict Weather"):
    # Prepare input (encoding categorical values)
    input_data = [
        Temperature,
        Humidity,
        Wind_Speed,
        Precipitation,
        ["Clear", "Partly Cloudy", "Overcast"].index(Cloud_Cover),
        Atmospheric_Pressure,
        UV_Index,
        ["Winter", "Spring", "Summer", "Autumn"].index(Season),
        Visibility,
        ["Urban", "Rural", "Coastal", "Mountain"].index(Location)
    ]

    reshaped_input = np.array(input_data).reshape(1, -1)
    prediction = model.predict(reshaped_input)

    st.success(f"✅ Prediction Result: {prediction[0]}")
