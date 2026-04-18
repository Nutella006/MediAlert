import numpy as np

def compute_hydration(row):


    hr = row["heart_rate"] / 120
    hrv = 1 - min(row["hrv"] / 100, 1)
    gsr = row["gsr"] / 800
    temp = abs(row["temperature"] - 36.5)


    steps = min(row["steps"] / 10000, 1)

    accel_mag = np.sqrt(
        row["acc_x"]**2 +
        row["acc_y"]**2 +
        row["acc_z"]**2
    )

    motion = min(accel_mag / 10, 1)

    sleep = 1 - row["sleep_score"]  

    blood_flow = 1 - row["blood_flow"] 

    dehydration_risk = (
        0.25 * hr +
        0.20 * hrv +
        0.15 * gsr +
        0.15 * temp +
        0.10 * steps +
        0.05 * motion +
        0.05 * sleep +
        0.05 * blood_flow
    )

    hydration = 1 - min(dehydration_risk, 1)

    return round(hydration * 100, 2)