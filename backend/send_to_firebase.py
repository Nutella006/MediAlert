import firebase_admin
from firebase_admin import credentials, db
from alerts import check_alerts
from ml_model import predict_health
import pandas as pd
import time


cred = credentials.Certificate("firebase_key.json")

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://medialert-c0f64-default-rtdb.firebaseio.com/'
})


ref = db.reference("health_data")


df = pd.read_csv("../data/health_data.csv")

print("Sending data to Firebase...\n")

for index, row in df.iterrows():
    data = {
        "time": int(row["time"]),
        "heart_rate": float(row["heart_rate"]),
        "spo2": float(row["spo2"]),
        "temperature": float(row["temperature"]),
        "steps": int(row["steps"]),
        "hrv": float(row["hrv"]),
        "stress": int(row["stress"])
    }

    alerts = check_alerts(data)

    # 🧠 ML Prediction
    health_status = predict_health(data)

    payload = {
        "data": data,
        "alerts": alerts,
        "prediction": health_status
    }

    ref.push(payload)

    print("Sent:", payload)

    time.sleep(1)