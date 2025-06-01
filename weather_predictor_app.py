import streamlit as st
import numpy as np
import joblib

# Load models and encoders
reg_model = joblib.load("weather_model.pkl")
cloud_model = joblib.load("cloud_model.pkl")
scaler = joblib.load("scaler.pkl")
cloud_encoder = joblib.load("cloud_encoder.pkl")

st.title("🌤️ Weather Predictor for Tomorrow")

st.markdown("Enter the past **3 days** of weather data to predict the **next day's** weather.")

cloud_options = list(cloud_encoder.classes_)

def day_input(day_num):
    st.subheader(f"Day {day_num}")
    temp = st.number_input(f"Temperature (°C) - Day {day_num}", 0.0, 60.0, 30.0)
    hum = st.number_input(f"Humidity (%) - Day {day_num}", 0.0, 100.0, 50.0)
    rain = st.number_input(f"Precipitation (%) - Day {day_num}", 0.0, 100.0, 10.0)
    wind = st.number_input(f"Wind Speed (kph) - Day {day_num}", 0.0, 100.0, 15.0)
    cloud = st.selectbox(f"Cloud Cover - Day {day_num}", cloud_options, key=f"cloud{day_num}")
    return [temp, hum, rain, wind, cloud_encoder.transform([cloud])[0]]

# Collect inputs for all 3 days
inputs = []
for i in range(1, 4):
    inputs.extend(day_input(i))

# Predict
if st.button("🔮 Predict Weather for Tomorrow"):
    X_input = np.array([inputs])
    X_scaled = scaler.transform(X_input)

    # Predict numeric
    temp_pred, hum_pred, rain_pred, wind_pred = reg_model.predict(X_scaled)[0]

    # Predict cloud
    cloud_pred_encoded = cloud_model.predict(X_scaled)[0]
    cloud_pred = cloud_encoder.inverse_transform([cloud_pred_encoded])[0]

    st.success("✅ Prediction Complete!")
    st.subheader("📅 Predicted Weather for Tomorrow:")
    st.markdown(f"- 🌡️ **Temperature**: `{temp_pred:.2f} °C`")
    st.markdown(f"- 💧 **Humidity**: `{hum_pred:.2f} %`")
    st.markdown(f"- ☔ **Precipitation**: `{rain_pred:.2f} %`")
    st.markdown(f"- 🌬️ **Wind Speed**: `{wind_pred:.2f} kph`")
    st.markdown(f"- ☁️ **Cloud Cover**: `{cloud_pred}`")
