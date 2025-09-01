import pandas as pd
import pickle as pk
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# ==========================
# Load the saved pipeline model (with preprocessing)
# ==========================
model = pk.load(open("weather.sav", "rb"))

# Sidebar for navigation
st.sidebar.title("Navigation")
selected = st.sidebar.selectbox("Choose the prediction model", ["Weather Forecasting"])

# ==========================
# Weather Forecasting Page
# ==========================
if selected == "Weather Forecasting":
    st.markdown("<h1 style='text-decoration: underline;'>Weather Forecasting using ML</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input("🌡 Temperature (°C)", -50.0, 60.0, 25.0)
        wind_speed = st.number_input("💨 Wind Speed (km/h)", 0.0, 200.0, 10.0)
        cloud_cover = st.selectbox("☁️ Cloud Cover", ["clear", "partly cloudy", "cloudy", "overcast"])
        uv_index = st.number_input("☀️ UV Index", 0.0, 15.0, 5.0)
        visibility = st.number_input("👀 Visibility (km)", 0.0, 50.0, 10.0)

    with col2:
        humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, 50.0)
        precipitation = st.number_input("🌧 Precipitation (%)", 0.0, 100.0, 10.0)
        pressure = st.number_input("🌬 Atmospheric Pressure (hPa)", 800.0, 1100.0, 1013.0)
        season = st.selectbox("🍂 Season", ["Autumn", "Spring", "Summer", "Winter"])
        location = st.selectbox("📍 Location", ["coastal", "inland", "mountain"])

    # Prediction
    if st.button("🔮 Predict Weather"):
        try:
            # Make a DataFrame (feature names must match training data)
            input_df = pd.DataFrame([{
                "Temperature": temperature,
                "Humidity": humidity,
                "Wind Speed": wind_speed,
                "Precipitation (%)": precipitation,
                "Cloud Cover": cloud_cover,
                "Atmospheric Pressure": pressure,
                "UV Index": uv_index,
                "Visibility (km)": visibility,
                "Season": season,
                "Location": location
            }])

            # Predict
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]

            # Display result
            st.success(f"🌤 Predicted Weather Category: **{pred}**")

            # Probability distribution
            st.subheader("📊 Prediction Probabilities")
            prob_df = pd.DataFrame({
                "Weather Category": model.classes_,
                "Probability (%)": [round(p * 100, 2) for p in prob]
            })
            st.table(prob_df)

            # Pie chart
            st.subheader("🔵 Probability Distribution (Pie Chart)")
            st.pyplot(prob_df.set_index("Weather Category").plot.pie(
                y="Probability (%)", autopct="%.1f%%", figsize=(5,5)).figure)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
