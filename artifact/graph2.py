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
instr_large    = np.array([0.8, 12, 8], dtype=float)

# Small workload data
baseline_small = np.array([1, 1, 1], dtype=float)
overhead_small = np.array([1, 1, 20], dtype=float)
instr_small    = np.array([0.8, 12, 8], dtype=float)

# Compute percentages
def pct_parts(b, o, i):
    total = b + o + i
    return b/total*100, o/total*100, i/total*100

baseline_pct_large, overhead_pct_large, instr_pct_large = pct_parts(baseline_large, overhead_large, instr_large)
baseline_pct_small, overhead_pct_small, instr_pct_small = pct_parts(baseline_small, overhead_small, instr_small)

# Create combined figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Plot large workload percentages
x = np.arange(len(techniques))
width = 0.6
ax1.bar(x,               baseline_pct_large, width, label='Baseline')
ax1.bar(x, overhead_pct_large, width, bottom=baseline_pct_large, label='Overhead')
ax1.bar(x,    instr_pct_large, width, bottom=baseline_pct_large+overhead_pct_large, label='Instrumentation')
ax1.set_ylim(0, 100)

# Plot small workload percentages
ax2.bar(x,               baseline_pct_small, width, label='Baseline')
ax2.bar(x, overhead_pct_small, width, bottom=baseline_pct_small, label='Overhead')
ax2.bar(x,    instr_pct_small, width, bottom=baseline_pct_small+overhead_pct_small, label='Instrumentation')
ax2.set_ylim(0, 100)

# 把子图的标题去掉，改为 x 轴标签
ax1.set_title('')
ax2.set_title('')
ax1.set_xticks(x); ax1.set_xticklabels(techniques)
ax2.set_xticks(x); ax2.set_xticklabels(techniques)
ax1.set_xlabel('Large Workload')
ax2.set_xlabel('Small Workload')
ax1.set_ylabel('Percentage (%)')

# 将图例移动到画布顶部中央
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02))

plt.tight_layout()
plt.savefig('motivation_swapped.pdf', bbox_inches='tight')
