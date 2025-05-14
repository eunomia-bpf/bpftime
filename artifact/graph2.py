import matplotlib.pyplot as plt
import numpy as np

# Set font size
font = {'size': 15}
plt.rc('font', **font)

# Techniques and data
techniques = ['Cuprobe', 'CUPTI', 'NVBit']

# Large workload data
baseline_large = np.array([100, 100, 100], dtype=float)
overhead_large = np.array([5, 5, 100], dtype=float)
instr_large = np.array([0.8, 12, 8], dtype=float)

# Small workload data
baseline_small = np.array([1, 1, 1], dtype=float)
overhead_small = np.array([1, 1, 20], dtype=float)
instr_small = np.array([0.8, 12, 8], dtype=float)

# Compute percentages
total_large = baseline_large + overhead_large + instr_large
baseline_pct_large = baseline_large / total_large * 100
overhead_pct_large = overhead_large / total_large * 100
instr_pct_large = instr_large / total_large * 100

total_small = baseline_small + overhead_small + instr_small
baseline_pct_small = baseline_small / total_small * 100
overhead_pct_small = overhead_small / total_small * 100
instr_pct_small = instr_small / total_small * 100

# Create combined figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Plot large workload percentages
ax1.bar(techniques, baseline_pct_large, label='Baseline')
ax1.bar(techniques, overhead_pct_large, bottom=baseline_pct_large, label='Overhead')
ax1.bar(techniques, instr_pct_large, bottom=baseline_pct_large + overhead_pct_large, label='Instrumentation')
ax1.set_ylabel('Percentage (%)')
ax1.set_title('Large Workload')
ax1.set_ylim(0, 100)

# Plot small workload percentages
ax2.bar(techniques, baseline_pct_small, label='Baseline')
ax2.bar(techniques, overhead_pct_small, bottom=baseline_pct_small, label='Overhead')
ax2.bar(techniques, instr_pct_small, bottom=baseline_pct_small + overhead_pct_small, label='Instrumentation')
ax2.set_title('Small Workload')
ax2.set_ylim(0, 100)

# Common legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig('motivation.pdf', bbox_inches='tight')