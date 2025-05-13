import matplotlib.pyplot as plt

# Latency metrics data
metrics = ['p10', 'p50', 'p99', 'mean']
native = [129.0, 129.0, 158.2299999999998, 130.234375]
profiler = [135.0, 136.0, 189.66999999999976, 138.84375]
egpu = [133.0, 139.0, 192.66999999999976, 137.15632]

# Plotting
x = range(len(metrics))
width = 0.25

plt.figure()
plt.bar([i - width for i in x], native, width=width, label='native')
plt.bar(x, profiler, width=width, label='pytorch-profiler')
plt.bar([i + width for i in x], egpu, width=width, label='egpu')
plt.xticks(x, metrics)
plt.ylabel('Latency (ms)')
plt.title('Latency Metrics Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('latency_comparison.pdf')