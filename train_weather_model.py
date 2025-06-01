import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Generate synthetic data
def generate_data(n_samples=1000):
    np.random.seed(42)
    data = []

    cloud_categories = ['Clear', 'Partly Cloudy', 'Cloudy']

    for _ in range(n_samples):
        t1 = np.random.uniform(10, 35)
        h1 = np.random.uniform(20, 100)
        r1 = np.random.uniform(0, 100)
        w1 = np.random.uniform(0, 40)
        c1 = np.random.choice(cloud_categories)

        t2 = t1 + np.random.normal(0, 2)
        h2 = h1 + np.random.normal(0, 5)
        r2 = r1 + np.random.normal(0, 10)
        w2 = w1 + np.random.normal(0, 5)
        c2 = np.random.choice(cloud_categories)

        t3 = t2 + np.random.normal(0, 2)
        h3 = h2 + np.random.normal(0, 5)
        r3 = r2 + np.random.normal(0, 10)
        w3 = w2 + np.random.normal(0, 5)
        c3 = np.random.choice(cloud_categories)

        # Day 4 is based on trend + noise
        t4 = t3 + np.random.normal(0, 2)
        h4 = h3 + np.random.normal(0, 5)
        r4 = r3 + np.random.normal(0, 10)
        w4 = w3 + np.random.normal(0, 5)
        c4 = np.random.choice(cloud_categories)

        data.append([t1, h1, r1, w1, c1, t2, h2, r2, w2, c2, t3, h3, r3, w3, c3, t4, h4, r4, w4, c4])

    columns = [
        'temp1', 'hum1', 'rain1', 'wind1', 'cloud1',
        'temp2', 'hum2', 'rain2', 'wind2', 'cloud2',
        'temp3', 'hum3', 'rain3', 'wind3', 'cloud3',
        'temp4', 'hum4', 'rain4', 'wind4', 'cloud4'
    ]

    return pd.DataFrame(data, columns=columns)

# Prepare and train models
df = generate_data()

# Encode cloud cover
cloud_encoder = LabelEncoder()
for col in ['cloud1', 'cloud2', 'cloud3', 'cloud4']:
    df[col] = cloud_encoder.fit_transform(df[col])

# Features and targets
X = df[['temp1', 'hum1', 'rain1', 'wind1', 'cloud1',
        'temp2', 'hum2', 'rain2', 'wind2', 'cloud2',
        'temp3', 'hum3', 'rain3', 'wind3', 'cloud3']]

y_temp = df['temp4']
y_hum = df['hum4']
y_rain = df['rain4']
y_wind = df['wind4']
y_cloud = df['cloud4']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Regression model for numeric values
reg_model = LinearRegression()
reg_model.fit(X_scaled, np.column_stack((y_temp, y_hum, y_rain, y_wind)))

# Classification model for cloud
cloud_model = RandomForestClassifier()
cloud_model.fit(X_scaled, y_cloud)

# Save models
joblib.dump(reg_model, 'weather_model.pkl')
joblib.dump(cloud_model, 'cloud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(cloud_encoder, 'cloud_encoder.pkl')

print("✅ Models trained and saved.")
