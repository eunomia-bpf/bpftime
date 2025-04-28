# BPFtime Uprobe Benchmark Results

*Generated on 2025-04-28 15:14:04*

## Environment

- **OS:** Linux 6.11.0-24-generic
- **CPU:** Intel(R) Core(TM) Ultra 7 258V (4 cores, 4 threads)
- **Memory:** 15.07 GB
- **Python:** 3.12.7

## Summary

This benchmark compares three different eBPF execution environments:
- **Kernel Uprobe**: Traditional kernel-based eBPF uprobes
- **Userspace Uprobe**: BPFtime's userspace implementation of uprobes
- **Embedded VM**: BPFtime's embedded eBPF VM

*Times shown in nanoseconds (ns) - lower is better*

### Core Uprobe Performance

| Operation | Kernel Uprobe | Userspace Uprobe | Speedup |
|-----------|---------------|------------------|---------|
| __bench_uprobe | 2177.25 | 172.13 | 12.65x |
| __bench_uretprobe | 2623.65 | 178.68 | 14.68x |
| __bench_uprobe_uretprobe | 2926.97 | 215.85 | 13.56x |

### Kernel eBPF Performance

| Operation | Min (ns) | Max (ns) | Avg (ns) | Std Dev |
|-----------|----------|----------|----------|---------|
| __bench_uprobe_uretprobe | 2926.97 | 2926.97 | 2926.97 | 0.00 |
| __bench_uretprobe | 2623.65 | 2623.65 | 2623.65 | 0.00 |
| __bench_uprobe | 2177.25 | 2177.25 | 2177.25 | 0.00 |
| __bench_read | 22223.99 | 22223.99 | 22223.99 | 0.00 |
| __bench_write | 23213.06 | 23213.06 | 23213.06 | 0.00 |
| __bench_hash_map_update | 41505.07 | 41505.07 | 41505.07 | 0.00 |
| __bench_hash_map_lookup | 12789.52 | 12789.52 | 12789.52 | 0.00 |
| __bench_hash_map_delete | 19354.34 | 19354.34 | 19354.34 | 0.00 |
| __bench_array_map_update | 11318.11 | 11318.11 | 11318.11 | 0.00 |
| __bench_array_map_lookup | 2784.35 | 2784.35 | 2784.35 | 0.00 |
| __bench_array_map_delete | 2740.89 | 2740.89 | 2740.89 | 0.00 |
| __bench_per_cpu_hash_map_update | 27439.99 | 27439.99 | 27439.99 | 0.00 |
| __bench_per_cpu_hash_map_lookup | 10150.70 | 10150.70 | 10150.70 | 0.00 |
| __bench_per_cpu_hash_map_delete | 21132.74 | 21132.74 | 21132.74 | 0.00 |
| __bench_per_cpu_array_map_update | 11052.69 | 11052.69 | 11052.69 | 0.00 |
| __bench_per_cpu_array_map_lookup | 3178.38 | 3178.38 | 3178.38 | 0.00 |
| __bench_per_cpu_array_map_delete | 3070.67 | 3070.67 | 3070.67 | 0.00 |

### Userspace eBPF Performance

| Operation | Min (ns) | Max (ns) | Avg (ns) | Std Dev |
|-----------|----------|----------|----------|---------|
| __bench_uprobe_uretprobe | 215.85 | 215.85 | 215.85 | 0.00 |
| __bench_uretprobe | 178.68 | 178.68 | 178.68 | 0.00 |
| __bench_uprobe | 172.13 | 172.13 | 172.13 | 0.00 |
| __bench_read | 10321.81 | 10321.81 | 10321.81 | 0.00 |
| __bench_write | 13454.07 | 13454.07 | 13454.07 | 0.00 |
| __bench_hash_map_update | 36364.03 | 36364.03 | 36364.03 | 0.00 |
| __bench_hash_map_lookup | 34287.46 | 34287.46 | 34287.46 | 0.00 |
| __bench_hash_map_delete | 18960.36 | 18960.36 | 18960.36 | 0.00 |
| __bench_array_map_update | 16387.31 | 16387.31 | 16387.31 | 0.00 |
| __bench_array_map_lookup | 14648.72 | 14648.72 | 14648.72 | 0.00 |
| __bench_array_map_delete | 13862.74 | 13862.74 | 13862.74 | 0.00 |
| __bench_per_cpu_hash_map_update | 80354.05 | 80354.05 | 80354.05 | 0.00 |
| __bench_per_cpu_hash_map_lookup | 63585.30 | 63585.30 | 63585.30 | 0.00 |
| __bench_per_cpu_hash_map_delete | 71262.50 | 71262.50 | 71262.50 | 0.00 |
| __bench_per_cpu_array_map_update | 26934.78 | 26934.78 | 26934.78 | 0.00 |
| __bench_per_cpu_array_map_lookup | 17591.17 | 17591.17 | 17591.17 | 0.00 |
| __bench_per_cpu_array_map_delete | 13325.91 | 13325.91 | 13325.91 | 0.00 |

### Embedded VM Performance

| Operation | Min (ns) | Max (ns) | Avg (ns) | Std Dev |
|-----------|----------|----------|----------|---------|
| embed | ∞ | ∞ | ∞ | 0.00 |

## Benchmark Metadata

- **Number of runs:** 1
- **Timestamp:** 2025-04-28 15:14:04
- **Total duration:** 76.31 seconds


## Notes

⚠️ The embedded VM benchmark reported infinity values, which indicates failures or timeouts.