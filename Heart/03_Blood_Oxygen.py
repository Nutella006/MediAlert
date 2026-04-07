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

plt.figure(figsize=(14, 5))
plt.plot(sample_indices, bpm_per_sample, color='crimson', linewidth=1.5)
plt.axhline(np.mean(bpm_per_sample), color='gray', linestyle='--', 
            label=f'Avg: {np.mean(bpm_per_sample):.1f} BPM')

plt.title('Heart Rate — Every Sample Point')
plt.xlabel('Sample Index (1, 2, 3 ...)')
plt.ylabel('Heart Rate (BPM)')
plt.ylim(40, 180)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


df['HR_BPM'] = bpm_per_sample
print(df[['IR_PPG', 'Red_PPG', 'HR_BPM']].head(20))
