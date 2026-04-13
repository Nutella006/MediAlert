def check_alerts(data):
    alerts = []


    if data["heart_rate"] > 100:
        alerts.append(" High Heart Rate")
    elif data["heart_rate"] < 50:
        alerts.append(" Low Heart Rate")


    if data["spo2"] < 94:
        alerts.append(" Low Oxygen Level")


    if data["temperature"] > 37.5:
        alerts.append(" High Body Temperature")


    if data["stress"] > 70:
        alerts.append(" High Stress Level")

    return alerts