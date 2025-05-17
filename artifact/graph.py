import matplotlib.pyplot as plt
font = {'size': 20}
plt.rc('font', **font)
# 设置更扁平的画布：宽 25，高 3
plt.figure(figsize=(15, 6))

# Latency metrics data
metrics = ['p10', 'p50', 'p99', 'mean']
native = [129.0, 129.0, 158.23, 130.23]
native_std = [1.2, 0.5, 3.2, 2.2]
profiler = [135.0, 136.0, 189.67, 138.84]
profiler_std = [1.3, 1.3, 1.3, 1.3]
egpu = [133.0, 139.0, 192.67, 137.16]
egpu_std = [0.9, 0.3, 0.8, 1.1]

x = range(len(metrics))
width = 0.25

plt.bar([i - width for i in x], native,  width=width, label='native',          yerr=native_std)
plt.bar( x,                   profiler, width=width, label='pytorch-profiler', yerr=profiler_std)
plt.bar([i + width for i in x], egpu,   width=width, label='cuprobe',         yerr=egpu_std)

plt.xticks(x, metrics)
plt.ylabel('Latency (ms)')
plt.title('Latency Metrics Comparison')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('latency_comparison.pdf', dpi=300)