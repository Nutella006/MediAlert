import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

df = pd.read_csv('ppg_subject_001.csv')
ir = df['IR_PPG'].values

fs = 100

def bandpass_filter(signal, lowcut=0.5, highcut=4.0, fs=100, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

filtered = bandpass_filter(ir, fs=fs)

peaks, _ = find_peaks(filtered, distance=fs*0.4, height=0)


rr_intervals = np.diff(peaks) / fs
bpm_per_beat = 60 / rr_intervals
mid_peaks = (peaks[:-1] + peaks[1:]) / 2 


sample_indices = np.arange(len(ir))        
bpm_per_sample = np.interp(sample_indices, mid_peaks, bpm_per_beat)


sdnn = np.std(rr_intervals)


rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))

diff_rr = np.abs(np.diff(rr_intervals))
nn50 = np.sum(diff_rr > 0.05)  
pnn50 = (nn50 / len(diff_rr)) * 100

print("\n===== HRV METRICS =====")
print(f"SDNN  : {sdnn:.4f} sec")
print(f"RMSSD : {rmssd:.4f} sec")
print(f"pNN50 : {pnn50:.2f} %")

plt.figure(figsize=(10,4))
plt.plot(rr_intervals, marker='o')
plt.title("RR Intervals (Heart Beat Timing)")
plt.xlabel("Beat Number")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.show()