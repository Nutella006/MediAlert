from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient

from feature_service import process_features

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

client = MongoClient("mongodb://localhost:27017/")
db = client["iotDB"]
collection = db["sensordata"]



def safe_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]



@app.get("/dashboard/latest")
def get_latest():

    doc = collection.find_one(sort=[("_id", -1)])

    if not doc:
        return JSONResponse(
            content={"error": "No data found"},
            status_code=404
        )

    features = process_features(doc)

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
        "sleep_state": features.get("sleep_state")
    },
    headers={
        "Cache-Control": "no-cache, no-store, must-revalidate"
    }
)


ir_buffer = []
red_buffer = []

accel_buffer = []
gyro_buffer = []

MAX_BUFFER = 100

@app.post("/sensor")
def receive_sensor(data: dict):

    global ir_buffer, red_buffer
    global accel_buffer, gyro_buffer


    ir = data.get("ir")
    red = data.get("red")

    acc_x = data.get("acc_x")
    acc_y = data.get("acc_y")
    acc_z = data.get("acc_z")

    gyro_x = data.get("gyro_x")
    gyro_y = data.get("gyro_y")
    gyro_z = data.get("gyro_z")


    if ir is not None:
        ir_buffer.append(ir)
    if red is not None:
        red_buffer.append(red)

    if None not in [acc_x, acc_y, acc_z]:
        accel_buffer.append((acc_x, acc_y, acc_z))

    if None not in [gyro_x, gyro_y, gyro_z]:
        gyro_buffer.append((gyro_x, gyro_y, gyro_z))



    if len(ir_buffer) > MAX_BUFFER:
        ir_buffer.pop(0)

    if len(red_buffer) > MAX_BUFFER:
        red_buffer.pop(0)

    if len(accel_buffer) > MAX_BUFFER:
        accel_buffer.pop(0)

    if len(gyro_buffer) > MAX_BUFFER:
        gyro_buffer.pop(0)

 

    doc = {
        "timestamp": data.get("timestamp"),

        "ir_signal": ir_buffer,
        "red_signal": red_buffer,

        "accel_series": accel_buffer,
        "gyro_series": gyro_buffer,

        "spo2": data.get("spo2"),
        "temperature": data.get("temp")
    }

    features = process_features(doc)
    doc.update(features)

    collection.insert_one(doc)

    print("BUFFER SIZE:", len(ir_buffer))

    return {"status": "ok"}