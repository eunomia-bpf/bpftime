import matplotlib.pyplot as plt
font = {'size': 15}
plt.rc('font', **font)
# Latency metrics data
metrics = ['p10', 'p50', 'p99', 'mean']
native = [129.0, 129.0, 158.2299999999998, 130.234375]
native_std = [1.2, 0.5, 3.2, 2.2]
profiler = [135.0, 136.0, 189.66999999999976, 138.84375]
profiler_std = [1.3, 1.3, 1.3, 1.3]
egpu = [133.0, 139.0, 192.66999999999976, 137.15632]
egpu_std = [0.9, 0.3, 0.8, 1.1]

# Plotting
x = range(len(metrics))
width = 0.25

plt.figure()
plt.bar([i - width for i in x], native, width=width, label='native', yerr=native_std)
plt.bar(x, profiler, width=width, label='pytorch-profiler', yerr=profiler_std)
plt.bar([i + width for i in x], egpu, width=width, label='cuprobe', yerr=egpu_std)
plt.xticks(x, metrics)
plt.ylabel('Latency (ms)')
plt.title('Latency Metrics Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('latency_comparison.pdf')