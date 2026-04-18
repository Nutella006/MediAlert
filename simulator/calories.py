import numpy as np
def compute_calories(row):

    hr = row["heart_rate"] / 120
    steps = min(row["steps"] / 10000, 1)

    motion = min(
        np.sqrt(row["acc_x"]**2 + row["acc_y"]**2 + row["acc_z"]**2) / 10,
        1
    )

    temp_factor = abs(row["temperature"] - 36.5) / 2

    gsr = row["gsr"] / 800

    calories = (
        0.35 * hr +
        0.35 * steps +
        0.15 * motion +
        0.10 * temp_factor +
        0.05 * gsr
    )

    return round(calories * 600, 2)