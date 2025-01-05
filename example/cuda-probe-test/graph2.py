import matplotlib.pyplot as plt
import numpy as np  # Add NumPy import

# Data for the chart
techniques = ['Static PTX', 'CUPTI + NVTX', 'NVBit']
baseline_runtime = [100, 100, 100]  # baseline execution time in µs
overhead_runtime = [5, 5, 100]  # baseline execution time in µs
instrumentation_overhead = [0.8, 12, 8]  # instrumentation overhead in µs

plt.figure(figsize=(6, 4))
# Plot baseline and overhead segments
plt.bar(techniques, baseline_runtime, label='Baseline Runtime')
plt.bar(techniques, overhead_runtime, bottom=baseline_runtime, label='Overhead Runtime')
# Convert lists to NumPy arrays for element-wise addition
baseline_array = np.array(baseline_runtime)
overhead_array = np.array(overhead_runtime)
plt.bar(techniques, instrumentation_overhead, bottom=baseline_array + overhead_array, label='Instrumentation Overhead')

plt.ylabel('Time (µs)')
plt.legend()

# Annotate segments
for i in r  ange(len(techniques)):
    plt.text(i, baseline_runtime[i]/2, f'{baseline_runtime[i]} µs', ha='center', va='center')
    plt.text(i, baseline_runtime[i] + overhead_runtime[i] + instrumentation_overhead[i]/2, f'{instrumentation_overhead[i]} µs', ha='center', va='center')

plt.tight_layout()
plt.savefig('motivation.pdf')