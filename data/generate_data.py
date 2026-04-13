import pandas as pd
import numpy as np

n = 200

time = np.arange(n)

heart_rate = np.random.normal(75, 5, n)  
spo2 = np.random.normal(98, 1, n)       
temperature = np.random.normal(36.8, 0.3, n)
steps = np.random.randint(0, 10, n)


hrv = np.random.normal(50, 10, n)

stress = np.random.randint(20, 80, n)

df = pd.DataFrame({
    "time": time,
    "heart_rate": heart_rate,
    "spo2": spo2,
    "temperature": temperature,
    "steps": steps,
    "hrv": hrv,
    "stress": stress
})

df.to_csv("health_data.csv", index=False)

print("Data generated successfully!")