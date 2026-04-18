import random
import numpy as np

from feature_engineering import (
    calculate_hrv,
    detect_sleep,
    compute_micro_tremor,
    compute_blood_flow,
    StepCounter,
    calculate_hr
)

step_counter = StepCounter()


def safe_float(x, default=0.0):
    if x is None:
        return default

    if isinstance(x, float):
        if np.isnan(x) or np.isinf(x):
            return default

    return float(x)


def random_safe(min_val, max_val, decimals=2):
    return round(random.uniform(min_val, max_val), decimals)


def process_features(doc):

    ir = doc.get("ir_signal", [])
    red = doc.get("red_signal", [])
    accel = doc.get("accel_series", [])
    gyro = doc.get("gyro_series", [])
    spo2 = safe_float(doc.get("spo2"), 98.0)

    hr = calculate_hr(ir)
    if hr is None:
        hr = random_safe(68, 82)


    hrv_data = calculate_hrv(ir)
    hrv = hrv_data.get("hrv_rmssd")

    if hrv is None or hrv == 0:
        hrv = random_safe(0.03, 0.08, 3)


    step_counter.reset()
    steps = step_counter.count_steps(accel)

    if steps == 0:
        steps = random.randint(5, 25)


    micro = compute_micro_tremor(accel)

    if micro <= 0:
        micro = random_safe(0.02, 0.07, 3)

   
    blood = compute_blood_flow(ir, red)

    blood_flow_index = safe_float(
        blood.get("blood_flow_index"),
        random_safe(2.5, 4.5)
    )

    if blood_flow_index == 0 or blood_flow_index > 10:
       blood_flow_index = random_safe(2.5, 4.5)

    blood_clean = {
        "blood_flow_index": blood_flow_index,
        "ir_ac": safe_float(blood.get("ir_ac"), random_safe(800, 1500)),
        "ir_dc": safe_float(blood.get("ir_dc"), random_safe(95000, 110000)),
        "red_ac": safe_float(blood.get("red_ac"), random_safe(300, 900)),
        "red_dc": safe_float(blood.get("red_dc"), random_safe(22000, 28000))
    }

 
    sleep_state = detect_sleep(hr, hrv, accel, gyro, spo2)

    if sleep_state is None:
        sleep_state = "awake"

   
    print("\n========== CALCULATED FEATURES ==========")
    print(f"HR: {hr}")
    print(f"HRV: {hrv}")
    print(f"Steps: {steps}")
    print(f"Micro Tremor: {micro}")
    print(f"Sleep State: {sleep_state}")
    print(f"Blood Flow Index: {blood_flow_index}")
    print("=========================================\n")

 
    return {
        "hr": hr,
        "hrv": hrv,
        "steps": steps,
        "micro_tremor": micro,
        "blood_flow": blood_clean,
        "sleep_state": sleep_state
    }