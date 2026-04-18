import numpy as np
import math


def calculate_hr(ir_signal, sampling_rate=50):


    if len(ir_signal) < 50:
        return None

    peaks = []


    mean_val = np.mean(ir_signal)
    std_val = np.std(ir_signal)

    threshold = mean_val + 0.8 * std_val

    min_distance = int(0.6 * sampling_rate)

    last_peak = -999

    for i in range(2, len(ir_signal) - 2):

        if (
            ir_signal[i] > threshold and
            ir_signal[i] > ir_signal[i - 1] and
            ir_signal[i] > ir_signal[i + 1] and
            ir_signal[i] > ir_signal[i - 2] and
            ir_signal[i] > ir_signal[i + 2]
        ):


            if i - last_peak > min_distance:
                peaks.append(i)
                last_peak = i

    if len(peaks) < 2:
        return None

    rr_intervals = np.diff(peaks) / sampling_rate

    rr_intervals = [rr for rr in rr_intervals if rr > 0.4]

    if len(rr_intervals) == 0:
        return None

    avg_rr = np.mean(rr_intervals)

    if avg_rr == 0:
        return None

    bpm = 60 / avg_rr

    if bpm < 50 or bpm > 130:
        return None

    return round(float(bpm), 2)


def calculate_hrv(ir_signal, sampling_rate=50):
    if len(ir_signal) < 5:
        return {"hrv_rmssd": None, "rr_intervals": []}

    peaks = []
    threshold = np.mean(ir_signal) + 0.3 * np.std(ir_signal)

    for i in range(1, len(ir_signal) - 1):
        if (
            ir_signal[i] > threshold and
            ir_signal[i] > ir_signal[i - 1] and
            ir_signal[i] > ir_signal[i + 1]
        ):
            peaks.append(i)

    if len(peaks) < 3:
        return {"hrv_rmssd": None, "rr_intervals": []}

    rr_intervals = np.diff(peaks) / sampling_rate

    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2)) if len(diff_rr) > 0 else None

    return {
        "hrv_rmssd": float(rmssd) if rmssd is not None else None,
        "rr_intervals": rr_intervals.tolist()
    }



class StepCounter:
    def __init__(self, threshold=1.2, cooldown=0.3, sampling_rate=50):
        self.threshold = threshold
        self.cooldown = cooldown
        self.sampling_rate = sampling_rate
        self.last_step_index = -999
        self.steps = 0

    def reset(self):
        self.steps = 0
        self.last_step_index = -999

    def count_steps(self, accel_data):
        if len(accel_data) < 3:
            return self.steps

        magnitudes = [
            math.sqrt(x*x + y*y + z*z)
            for x, y, z in accel_data
        ]

        for i in range(1, len(magnitudes) - 1):
            if (
                magnitudes[i] > self.threshold and
                magnitudes[i] > magnitudes[i - 1] and
                magnitudes[i] > magnitudes[i + 1]
            ):
                if i - self.last_step_index > self.cooldown * self.sampling_rate:
                    self.steps += 1
                    self.last_step_index = i

        return self.steps


def detect_sleep(hr, hrv, accel_data, gyro_data, spo2):

    if len(accel_data) == 0 or len(gyro_data) == 0:
        return "awake"

    accel_mags = [math.sqrt(x*x + y*y + z*z) for x, y, z in accel_data]
    gyro_mags = [math.sqrt(x*x + y*y + z*z) for x, y, z in gyro_data]

    avg_motion = np.mean(accel_mags)
    motion_var = np.var(accel_mags)
    gyro_activity = np.mean(gyro_mags)

    is_still = avg_motion < 1.2 and motion_var < 0.02
    low_gyro = gyro_activity < 0.5
    low_hr = hr is not None and hr < 65
    good_hrv = hrv is not None and hrv > 0.03
    stable_spo2 = spo2 is not None and spo2 > 94

    if is_still and low_gyro and low_hr and good_hrv and stable_spo2:
        return "deep_sleep"
    elif is_still and low_hr:
        return "light_sleep"
    else:
        return "awake"



def compute_blood_flow(ir_signal, red_signal):

    ir = np.array(ir_signal)
    red = np.array(red_signal)

    if len(ir) == 0 or len(red) == 0:
        return {
            "blood_flow_index": 0,
            "ir_ac": 0,
            "ir_dc": 0,
            "red_ac": 0,
            "red_dc": 0
        }

    ir_dc = np.mean(ir)
    red_dc = np.mean(red)

    ir_ac = np.std(ir)
    red_ac = np.std(red)

    if ir_dc == 0:
        return {"blood_flow_index": 0}

    perfusion = (ir_ac / ir_dc) * 10
    red_factor = (red_ac / red_dc) * 100 if red_dc != 0 else 0

    return {
        "blood_flow_index": float((perfusion + red_factor) / 2),
        "ir_ac": float(ir_ac),
        "ir_dc": float(ir_dc),
        "red_ac": float(red_ac),
        "red_dc": float(red_dc)
    }

def compute_micro_tremor(accel_data):

    if len(accel_data) < 3:
        return 0.0

    mags = np.array([
        math.sqrt(x*x + y*y + z*z)
        for x, y, z in accel_data
    ])

    centered = mags - np.mean(mags)

    micro_energy = np.mean(np.abs(centered))
    variance = np.var(centered)

    return float(micro_energy * 0.7 + variance * 0.3)