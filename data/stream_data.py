import pandas as pd
import time


df = pd.read_csv("health_data.csv")

print("Starting live data stream...\n")


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

    print(data)

    time.sleep(0.5)