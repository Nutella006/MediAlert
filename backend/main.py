from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient

from hydration import compute_hydration
from ml_stress import load_model, predict_stress
from ml_calories import predict_calories
from feature_service import process_features

import pickle
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ================= DB =================
client = MongoClient("mongodb://localhost:27017/")
db = client["iotDB"]
collection = db["sensordata"]

# ================= MODELS =================
stress_model, stress_features = load_model("stress_model_v2.pkl")

with open("calorie_model.pkl", "rb") as f:
    saved = pickle.load(f)
    calorie_model = saved["model"]
    calorie_features = saved["features"]

# ================= BUFFERS =================
ir_buffer = []
red_buffer = []
accel_buffer = []
gyro_buffer = []

MAX_BUFFER = 100


# ================= DASHBOARD API =================
@app.get("/dashboard/latest")
def get_latest():

    doc = collection.find_one(sort=[("_id", -1)])

    if not doc:
        return JSONResponse(
            content={"error": "No data found"},
            status_code=404
        )

    features = process_features(doc)

    sensor_input = {
        "heart_rate": features.get("hr"),
        "temperature": doc.get("temperature", 36.5),
        "blood_flow": features.get("blood_flow", {}).get("blood_flow_index", 1.0),
        "steps": features.get("steps", 0),
        "sleep_state": 1,
        "hrv": features.get("hrv", 0.05),
        "micro_tremor": features.get("micro_tremor", 0.05),
        "accel_series": {"x": 0.1, "y": 0.1, "z": 0.98},
        "gyro_series": {"std_x": 0.1, "std_y": 0.1, "std_z": 0.1},
    }

    stress_result = predict_stress(
        stress_model,
        stress_features,
        sensor_input
    )

    calories = predict_calories(
        calorie_model,
        calorie_features,
        sensor_input
    )

    hydration = compute_hydration({
        "heart_rate": sensor_input["heart_rate"],
        "hrv": sensor_input["hrv"],
        "temperature": sensor_input["temperature"],
        "blood_flow": sensor_input["blood_flow"],
        "steps": sensor_input["steps"],
        "acc_x": 0.1,
        "acc_y": 0.1,
        "acc_z": 0.98,
        "sleep_score": 0.8
    })

    return JSONResponse(
        content={
            "timestamp": doc.get("timestamp"),

            "ir_signal": doc.get("ir_signal"),
            "red_signal": doc.get("red_signal"),

            "accel_series": doc.get("accel_series"),
            "gyro_series": doc.get("gyro_series"),

            "spo2": doc.get("spo2"),
            "temperature": doc.get("temperature"),

            "heart_rate": features.get("hr"),
            "hrv": features.get("hrv"),
            "steps": features.get("steps"),
            "micro_tremor": features.get("micro_tremor"),
            "blood_flow": features.get("blood_flow"),
            "sleep_state": features.get("sleep_state"),

            "stress_score": stress_result.get("label"),
            "calories_burned": round(calories, 2),
            "hydration": hydration,

            # 🔥 NEW FEATURES
            "buzzer_status": doc.get("buzzer_status", 0),
            "sos": doc.get("sos", 0),
        },
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate"
        }
    )

from datetime import datetime

@app.post("/alert")
def alert(data: dict):

    print("🚨 ALERT RECEIVED:", data)

    collection.insert_one({
        "type": "sos",
        "sos": 1,
        "timestamp": datetime.now()
    })

    return {"status": "ok", "message": "SOS received"}


# ================= SENSOR API =================
@app.post("/sensor")
def receive_sensor(data: dict):

    global ir_buffer, red_buffer, accel_buffer, gyro_buffer

    ir = data.get("ir")
    red = data.get("red")

    acc_x = data.get("acc_x", round(random.uniform(0.01, 0.20), 2))
    acc_y = data.get("acc_y", round(random.uniform(0.01, 0.20), 2))
    acc_z = data.get("acc_z", round(random.uniform(0.90, 1.10), 2))

    gyro_x = data.get("gyro_x", round(random.uniform(0.01, 0.15), 2))
    gyro_y = data.get("gyro_y", round(random.uniform(0.01, 0.15), 2))
    gyro_z = data.get("gyro_z", round(random.uniform(0.01, 0.15), 2))

    # ================= BUFFER UPDATE =================
    if ir is not None:
        ir_buffer.append(ir)
    if red is not None:
        red_buffer.append(red)

    accel_buffer.append((acc_x, acc_y, acc_z))
    gyro_buffer.append((gyro_x, gyro_y, gyro_z))

    # limit buffers
    if len(ir_buffer) > MAX_BUFFER:
        ir_buffer.pop(0)
    if len(red_buffer) > MAX_BUFFER:
        red_buffer.pop(0)
    if len(accel_buffer) > MAX_BUFFER:
        accel_buffer.pop(0)
    if len(gyro_buffer) > MAX_BUFFER:
        gyro_buffer.pop(0)

    # ================= DOCUMENT =================
    doc = {
        "timestamp": data.get("timestamp"),

        "ir_signal": ir_buffer,
        "red_signal": red_buffer,

        "accel_series": accel_buffer,
        "gyro_series": gyro_buffer,

        "spo2": data.get("spo2"),
        "temperature": data.get("temp"),

        # 🔥 NEW
        "buzzer_status": data.get("buzzer_status", 0),
        "sos": data.get("sos", 0)
    }

    # features
    features = process_features(doc)
    doc.update(features)

    collection.insert_one(doc)

    print("BUFFER SIZE:", len(ir_buffer))

    return {"status": "ok"}

