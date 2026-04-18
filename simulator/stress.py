import numpy as np

def compute_stress(row):

    hr = row["heart_rate"] / 120
    hrv = 1 - min(row["hrv"] / 100, 1)
    gsr = row["gsr"] / 800

    temp = abs(row["temperature"] - 36.5) / 2

    sleep = 1 - row["sleep_score"]

    blood_flow = 1 - row["blood_flow"]

    accel_mag = np.sqrt(
        row["acc_x"]**2 +
        row["acc_y"]**2 +
        row["acc_z"]**2
    )
    motion = min(accel_mag / 10, 1)

    gyro_mag = np.sqrt(
        row["gyro_x"]**2 +
        row["gyro_y"]**2 +
        row["gyro_z"]**2
    )
    tremor = min(gyro_mag / 5, 1)

    steps = min(row["steps"] / 10000, 1)

    spo2 = 1 - min(row["spo2"] / 100, 1)

    stress_score = (
        0.25 * hr +
        0.20 * hrv +
        0.15 * gsr +
        0.10 * temp +
        0.10 * sleep +
        0.05 * blood_flow +
        0.05 * motion +
        0.05 * tremor +
        0.05 * steps +
        0.05 * spo2
    )

    stress = min(stress_score * 100, 100)

    return round(stress, 2)