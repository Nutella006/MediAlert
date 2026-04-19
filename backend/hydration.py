import numpy as np

def compute_hydration(row):
    """
    Estimate hydration level (0–100%) from wearable sensor data.
    Higher score = better hydrated.
    """


    hr_norm = np.clip((row["heart_rate"] - 60) / (180 - 60), 0, 1)

    hrv_norm = 1 - np.clip(row["hrv"] / 100, 0, 1)


    temp_deviation = abs(row["temperature"] - 36.5)
    temp_norm = np.clip(temp_deviation / 2.0, 0, 1) 

    blood_flow_risk = 1 - np.clip(row["blood_flow"], 0, 1)

  
    steps_norm = np.clip(row["steps"] / 10000, 0, 1)

    accel_mag = np.sqrt(row["acc_x"]**2 + row["acc_y"]**2 + row["acc_z"]**2)
    motion_norm = np.clip(accel_mag / 15, 0, 1) 

    sleep_risk = 1 - np.clip(row["sleep_score"], 0, 1)


    dehydration_risk = (
        0.28 * hr_norm        
      + 0.22 * hrv_norm       
      + 0.18 * temp_norm      
      + 0.15 * blood_flow_risk 
      + 0.08 * steps_norm     
      + 0.05 * motion_norm    
      + 0.04 * sleep_risk      
    )

    hydration = 1 - np.clip(dehydration_risk, 0, 1)
    return round(hydration * 100, 2)