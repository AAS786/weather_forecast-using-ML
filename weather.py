import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# ==========================
# Load the saved model
# ==========================
loaded_model = pk.load(open('weather.sav','rb'))

# ==========================
# Define encoders (same mapping as training)
# ==========================
cloud_cover_mapping = {
    'clear': 0,
    'cloudy': 1,
    'overcast': 2,
    'partly cloudy': 3
}

season_mapping = {
    'Autumn': 0,
    'Spring': 1,
    'Summer': 2,
    'Winter': 3
}

location_mapping = {
    'coastal': 0,
    'inland': 1,
    'mountain': 2
}

# Weather classes (adjust according to your dataset)
class_labels = {
    0: "Cloudy",
    1: "Rainy",
    2: "Snowy",
    3: "Sunny"
}

# Sidebar for navigation
st.sidebar.title("Navigation")
selected = st.sidebar.selectbox("Choose a page", ["Weather Forecasting", "About App"])

# ==========================
# Weather Forecasting Page
# ==========================
if selected == 'Weather Forecasting':
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>🌤️ Weather Forecasting using Machine Learning 🌤️</h1>", 
        unsafe_allow_html=True
    )
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input('Temperature (°C)')
        wind_speed = st.number_input('Wind Speed (km/h)')
        cloud_cover = st.selectbox('Cloud Cover', list(cloud_cover_mapping.keys()))
        uv_index = st.number_input('UV Index')
        visibility = st.number_input('Visibility (km)')
    with col2:
        humidity = st.number_input('Humidity (%)')
        precipitation = st.number_input('Precipitation (mm)')
        pressure = st.number_input('Atmospheric Pressure (hPa)')
        season = st.selectbox('Season', list(season_mapping.keys()))
        location = st.selectbox('Location', list(location_mapping.keys()))

    # Prediction result
    if st.button('Predict Weather'):
        # Encode categorical values
        cloud_cover_encoded = cloud_cover_mapping[cloud_cover]
        season_encoded = season_mapping[season]
        location_encoded = location_mapping[location]

        # Convert inputs to float
        input_data = [
            float(temperature),
            float(humidity),
            float(wind_speed),
            float(pressure),
            cloud_cover_encoded,
            float(uv_index),
            float(visibility),
            float(precipitation),
            season_encoded,
            location_encoded
        ]
        
        # Make prediction
        weather_pred = loaded_model.predict([input_data])[0]
        weather_prob = loaded_model.predict_proba([input_data])[0]  # probabilities

        # ==========================
        # Show Prediction with Emoji & Balloons
        # ==========================
        predicted_category = class_labels.get(weather_pred, 'Unknown')

        # Weather emojis
        weather_emojis = {
            "Sunny": "☀️",
            "Rainy": "🌧️",
            "Cloudy": "☁️",
            "Snowy": "❄️"
        }

        emoji = weather_emojis.get(predicted_category, "🌤️")
        st.success(f"Predicted Weather: {emoji} **{predicted_category}**")

        # 🎈 Balloons animation
        st.balloons()

# ==========================
# About App Page
# ==========================
elif selected == "About App":
    st.markdown("## ℹ️ About this App")
    st.info("""
    This **Weather Forecasting App** uses a Machine Learning model to predict weather 
    conditions (Sunny, Cloudy, Rainy, Snowy) based on various environmental parameters.  
    
    **Features**:
    - User-friendly interface
    - Visual probability charts
    - Fun weather-specific animations (🌧 Snow, ☀️ Toast messages, etc.)
    """)
