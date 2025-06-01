# 🌤️ Weather Prediction

A simple Streamlit web app that predicts tomorrow's weather based on the past 3 days of weather data. The app uses machine learning models trained on synthetic data to forecast:
- Temperature (°C)
- Humidity (%)
- Precipitation (%)
- Wind Speed (kph)
- Cloud Cover (Clear/Partly Cloudy/Cloudy)

## Demo
<img width="702" alt="Screenshot 2025-06-01 at 10 26 12 PM" src="https://github.com/user-attachments/assets/fe758fc0-6672-4d57-89f7-4342c4c7ef84" />
<img width="777" alt="Screenshot 2025-06-01 at 10 22 42 PM" src="https://github.com/user-attachments/assets/3fcf4bcf-7064-4ff7-afea-fd1f899b0890" />
<img width="781" alt="Screenshot 2025-06-01 at 10 22 49 PM" src="https://github.com/user-attachments/assets/339f550e-7e53-4a96-b2b7-5467a1642b22" />
<img width="764" alt="result" src="https://github.com/user-attachments/assets/20223bdb-4c94-4748-8e83-4875fdb61877" />


## Features
- User-friendly web interface
- Input weather data for the past 3 days
- Predicts next day's weather conditions
- Instant results with clear visualization

## Getting Started

### Prerequisites
- Python 3.10+

### Installation
1. Clone this repository or download the source code.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
Start the Streamlit app with:
```bash
streamlit run weather_predictor_app.py
```
This will open the app in your default web browser.

## Project Structure
- `weather_predictor_app.py` — Main Streamlit application
- `weather_model.pkl`, `cloud_model.pkl`, `scaler.pkl`, `cloud_encoder.pkl` — Pre-trained model and encoder files
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation

## Notes
- The models are trained on synthetic data for demonstration purposes. For real-world use, retrain with actual weather data.
- If you want to retrain the models, you can use a script similar to the original `train_weather_model.py` (not included in this cleaned version).

