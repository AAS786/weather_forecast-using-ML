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
selected = st.sidebar.selectbox(
    "Choose a page", 
    ("Weather Forecasting", "About App", "ML Models Used", "Dataset Info", "How It Works", "Contact")
)


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
    conditions (☀️ Sunny, ☁️ Cloudy, 🌧️ Rainy, ❄️ Snowy) based on various environmental parameters 
    like temperature, humidity, wind speed, pressure, and more.
    """)

    st.markdown("### 🌟 Features")
    st.markdown("""
    - 🎛 **User-friendly interface** with simple input fields  
    - 📊 **Visual probability charts** to show model confidence  
    - 🎉 **Fun weather animations** (🌧️ rain, ❄️ snow, ☀️ sunny toast messages)  
    - 🌍 **Works anywhere** with customizable parameters  
    """)
    st.markdown("### 🌍 Use Cases")
    st.markdown("""
    - 👨‍🌾 **Agriculture:** Helps farmers predict weather for crop planning  
    - ✈️ **Travel:** Assists travelers in planning safe journeys  
    - 🏙️ **Smart Cities:** Improves urban weather response systems  
    - ⚡ **Renewable Energy:** Useful for solar and wind energy forecasting  
    """) 


    st.markdown("### 🌦️ Weather Conditions Explained")
    st.markdown("""
    **☀️ Sunny**  
    - Clear skies with little or no clouds  
    - High UV index, warmer temperatures  
    - Great for outdoor activities but stay hydrated!  

    **☁️ Cloudy**  
    - Mostly covered skies with gray or white clouds  
    - Mild temperatures, lower sunlight  
    - May precede rainy or stormy conditions  

    **🌧️ Rainy**  
    - Precipitation in the form of light to heavy rain  
    - Increased humidity, cooler temperatures  
    - Carry an umbrella and avoid slippery roads  

    **❄️ Snowy**  
    - Cold conditions with snowfall  
    - Reduced visibility and icy surfaces  
    - Wear warm clothes and take safety precautions while traveling  
    """)

    st.success("💡 Tip: Enter realistic weather values for more accurate predictions!")

elif selected == "ML Models Used":
    st.markdown("## 🤖 Machine Learning Models Used in this Project")

    st.info("Here are the different ML algorithms we tried and their characteristics:")

    st.markdown("""
    ### 1. Gaussian Naive Bayes (GaussianNB)  
    - Probabilistic classifier based on Bayes' theorem  
    - Assumes features are normally distributed  
    - Works well with small datasets and real-time predictions  

    ### 2. Decision Tree Classifier 🌳  
    - Splits data into branches based on feature values  
    - Easy to interpret and visualize  
    - Can overfit if not pruned properly  

    ### 3. Random Forest Classifier 🌲🌲  
    - Ensemble of multiple decision trees  
    - Reduces overfitting and improves accuracy  
    - Good for handling missing values and large datasets  

    ### 4. Logistic Regression 📈  
    - Simple linear model for classification  
    - Outputs probabilities between 0 and 1  
    - Performs well when features are linearly separable  

    ### 5. K-Nearest Neighbors (KNN) 👥  
    - Classifies based on the majority vote of neighbors  
    - Simple and intuitive, but slow with large datasets  
    - Works well with normalized numerical features  

    ### 6. Support Vector Machine (SVM) ⚡  
    - Finds the optimal hyperplane that separates classes  
    - Effective in high-dimensional spaces  
    - Can be slow on large datasets  

    ### 7. Gradient Boosting Classifier 🌟  
    - Ensemble method that builds trees sequentially  
    - Each tree corrects errors of the previous one  
    - High accuracy but can be slower to train  

    ### 8. XGBoost Classifier 🚀  
    - Optimized gradient boosting algorithm  
    - Very fast, efficient, and widely used in competitions  
    - Handles missing values and large datasets effectively  
    """)

    st.success("💡 These models were compared, and the best-performing one was saved as `weather.sav` for predictions.")

# ==========================
# Model Performance Page
# ==========================

# elif selected == "Model Performance":
#     st.markdown("## 📊 Model Performance Comparison")

#     st.info("We evaluated multiple models on accuracy, precision, recall, and F1-score. Below is a summary:")

#     performance_data = {
#         "Model": ["Naive Bayes", "Decision Tree", "Random Forest", "Logistic Regression",
#                   "KNN", "SVM", "Gradient Boosting", "XGBoost"],
#         "Accuracy (%)": [78, 82, 88, 80, 79, 85, 87, 90],
#         "F1-Score": [0.77, 0.81, 0.88, 0.80, 0.78, 0.84, 0.86, 0.91]
#     }

#     df = pd.DataFrame(performance_data)
#     st.table(df)

#     st.bar_chart(df.set_index("Model")["Accuracy (%)"])

# ==========================
# Dataset Info Page
# ==========================

elif selected == "Dataset Info":
    st.markdown("## 📂 Dataset Information")
    st.markdown("""
    - **Rows:** ~13,200 weather observations  
    - **Features:** Temperature, Humidity, Wind Speed, Pressure, Cloud Cover, Season, Location, etc.  
    - **Target Variable:** Weather Condition (Sunny, Cloudy, Rainy, Snowy)  
    - **Source:** [Kaggle - Weather Type Classification](https://www.kaggle.com/datasets/nikhil7280/weather-type-classification)


    This dataset was preprocessed with:
    - Missing value handling  
    - Feature encoding (categorical to numeric)  
    - Normalization of continuous variables  
    """)

# ==========================
# How It Works Page
# ==========================

elif selected == "How It Works":
    st.markdown("## 📚 How This App Works")
    st.markdown("""
    1. **Input Parameters:** You provide weather details (temperature, humidity, etc.)  
    2. **Preprocessing:** The inputs are encoded and scaled  
    3. **Model Prediction:** The trained ML model (`weather.sav`) predicts the class  
    4. **Output:** The app displays the predicted weather with probability distribution  
    5. **Visualization:** Probability chart and animations make results engaging 🎉  
    """)

# ==========================
# Contact Page
# ==========================

elif selected == "Contact":
    st.markdown("## 👨‍💻 Developer Info")

    st.markdown("""
    **Developed By:** Asif Shaikh  
    🏷 Role: Machine Learning Developer | Data Science Enthusiast  

    📧 Email: Shaikhasif694.@gmail.com  
    🌐 GitHub: [AAS786](https://github.com/AAS786)  
    🔗 LinkedIn: [AsifShaikh](https://www.linkedin.com/in/AsifShaikh)  
    🌎 Portfolio: [My Portfolio Website](https://aas786.github.io/My-Portfolio-website/)  
    """)

    st.markdown("### 🛠 Skills")
    st.markdown("""
    🐍 Python | 📊 Pandas | 🔢 NumPy | 🤖 Scikit-learn | 🌲 Random Forest | 🚀 XGBoost | 🌐 Streamlit  
    """)

    st.markdown("### 🤖 Built with AI Assistance")
    st.info("""
    This project was created with the help of **AI tools** like ChatGPT for:  
    - Debugging and optimizing Python code  
    - Designing an interactive and user-friendly UI with Streamlit  
    - Improving project documentation and explanations  
    - Adding creative ideas (animations, layouts, and presentation)  

    💡 I believe in **AI-assisted development** — using AI as a coding partner to speed up learning, improve productivity, and explore new ideas.
    """)

    st.markdown("### 🙏 Special Thanks")
    st.success("""
    - Streamlit (for the interactive UI)  
    - Scikit-learn, XGBoost (for ML models)  
    - ChatGPT (for guidance, debugging, and documentation)  
    """)



