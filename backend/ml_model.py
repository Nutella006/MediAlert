def predict_health(data, history=[]):
    hr = data["heart_rate"]
    spo2 = data["spo2"]
    stress = data["stress"]
    temp = data["temperature"]

    score = 0


    if hr > 110 or hr < 50:
        score += 3
    elif hr > 95:
        score += 1


    if spo2 < 90:
        score += 3
    elif spo2 < 95:
        score += 2


    if stress > 80:
        score += 2
    elif stress > 60:
        score += 1


    if temp > 38 or temp < 35:
        score += 2
    elif temp > 37:
        score += 1

    history.append(hr)

    if len(history) > 5:
        history.pop(0)

    if len(history) == 5:
        if history[-1] > history[0] + 10:
            score += 2   

    if score >= 6:
        return "CRITICAL"
    elif score >= 3:
        return "WARNING"
    else:
        return "SAFE"