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

# Weather classes
class_labels = {
    0: "Cloudy",
    1: "Rainy",
    2: "Snowy",
    3: "Sunny"
}

# ==========================
# Page Config & Custom CSS
# ==========================
st.set_page_config(page_title="Weather Forecasting App", page_icon="🌦", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #e0f7fa, #ffffff);
    }
    h1 {
        color: #2e7d32;
        text-align: center;
        font-size: 40px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .css-1d391kg {  /* Streamlit main container */
        background: rgba(255,255,255,0.8);
        border-radius: 15px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# Sidebar
# ==========================
st.sidebar.title("🌍 Navigation")
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
        temperature = st.number_input('🌡 Temperature (°C)')
        wind_speed = st.number_input('💨 Wind Speed (km/h)')
        cloud_cover = st.selectbox('☁️ Cloud Cover', list(cloud_cover_mapping.keys()))
        uv_index = st.number_input('🔆 UV Index')
        visibility = st.number_input('👀 Visibility (km)')
    with col2:
        humidity = st.number_input('💧 Humidity (%)')
        precipitation = st.number_input('🌧 Precipitation (mm)')
        pressure = st.number_input('🌬 Atmospheric Pressure (hPa)')
        season = st.selectbox('🍂 Season', list(season_mapping.keys()))
        location = st.selectbox('🗺 Location', list(location_mapping.keys()))

    # Prediction result
    if st.button('🔮 Predict Weather'):
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
        weather_prob = loaded_model.predict_proba([input_data])[0]

        # Predicted category
        predicted_category = class_labels.get(weather_pred, 'Unknown')
        weather_emojis = {
            "Sunny": "☀️",
            "Rainy": "🌧️",
            "Cloudy": "☁️",
            "Snowy": "❄️"
        }
        emoji = weather_emojis.get(predicted_category, "🌤️")

        st.success(f"🎯 Predicted Weather: {emoji} **{predicted_category}**")

        # Weather-specific effects
        if predicted_category == "Sunny":
            st.toast("☀️ It's a bright and sunny day!")
        elif predicted_category == "Rainy":
            st.toast("🌧️ Don't forget your umbrella!")
        elif predicted_category == "Snowy":
            st.snow()
        elif predicted_category == "Cloudy":
            st.toast("☁️ Looks like a cloudy day ahead!")

        # Show probabilities
        st.markdown("### 📊 Prediction Probabilities")
        prob_df = pd.DataFrame({
            "Weather Category": [class_labels[i] for i in range(len(weather_prob))],
            "Probability (%)": [round(p*100, 2) for p in weather_prob]
        })
        st.bar_chart(prob_df.set_index("Weather Category"))

# ==========================
# About Page
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
