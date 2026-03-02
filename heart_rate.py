import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("dataset_used/Heart_rate/Heart_rate_S15.csv", skiprows=2)
values = data.iloc[:, 0]
activities = {
    "NO_ACTIVITY":    [0, 666, 1199, 1647, 2546, 3786, 6301, 7746, 7924],
    "BASELINE":       [54],
    "STAIRS":         [779],
    "SOCCER":         [1292],
    "CYCLING":        [2140],
    "DRIVING":        [3000],
    "LUNCH":          [4200],
    "WALKING":        [5641],
    "WORKING":        [6542],
    "CLEAN_BASELINE": [7803],
}
fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(values, color='#e53935', linewidth=1.2)

labeled = set()
for act, times in activities.items():
    for t in times:
        color = '#bbb' if act == 'NO_ACTIVITY' else '#1e88e5'
        ax.axvline(x=t, color=color, linewidth=0.8,linestyle='--')
        if act not in labeled:
            ax.text(t + 15, max(values) - 2, act.replace("_", " "),
                    rotation=90, fontsize=7, color='#888', va='top')
            labeled.add(act)

ax.set_title("Heart Rate over Activities", fontsize=13, pad=10)
ax.set_xlabel("Sample Number")
ax.set_ylabel("Heart Rate (BPM)")
ax.set_xlim(0, len(values))

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("heart_rate_plot.png", dpi=150, bbox_inches='tight')
plt.show()
