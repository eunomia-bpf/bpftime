# BPFtime Uprobe Benchmark Results

*Generated on 2025-04-28 16:06:04*

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
| __bench_uprobe | 2416.31 | 233.23 | 10.36x |
| __bench_uretprobe | 2870.88 | 197.06 | 14.57x |
| __bench_uprobe_uretprobe | 3040.79 | 196.79 | 15.45x |

### Kernel eBPF Performance

| Operation | Min (ns) | Max (ns) | Avg (ns) | Std Dev |
|-----------|----------|----------|----------|---------|
| __bench_uprobe_uretprobe | 2634.01 | 3437.30 | 3040.79 | 243.78 |
| __bench_uretprobe | 2548.73 | 3152.64 | 2870.88 | 230.91 |
| __bench_uprobe | 1996.40 | 2819.68 | 2416.31 | 258.84 |
| __bench_read | 20579.99 | 27294.64 | 23473.15 | 1911.24 |
| __bench_write | 20784.80 | 29112.56 | 24698.55 | 2058.82 |
| __bench_hash_map_update | 41468.41 | 53731.06 | 48651.75 | 4050.62 |
| __bench_hash_map_lookup | 9374.40 | 13056.65 | 10926.05 | 909.83 |
| __bench_hash_map_delete | 18335.53 | 25395.61 | 20698.13 | 2162.40 |
| __bench_array_map_update | 9497.25 | 14364.52 | 11550.03 | 1269.28 |
| __bench_array_map_lookup | 2509.89 | 3193.74 | 2831.07 | 229.90 |
| __bench_array_map_delete | 2693.73 | 3300.11 | 2968.31 | 207.97 |
| __bench_per_cpu_hash_map_update | 27081.29 | 36724.47 | 32425.47 | 3094.37 |
| __bench_per_cpu_hash_map_lookup | 9211.39 | 11809.20 | 10928.61 | 882.25 |
| __bench_per_cpu_hash_map_delete | 17740.25 | 23080.19 | 20610.10 | 1672.48 |
| __bench_per_cpu_array_map_update | 9268.89 | 14245.44 | 11960.28 | 1451.85 |
| __bench_per_cpu_array_map_lookup | 2527.32 | 3294.53 | 3036.76 | 208.42 |
| __bench_per_cpu_array_map_delete | 2644.95 | 4024.94 | 3222.00 | 338.45 |

### Userspace eBPF Performance

| Operation | Min (ns) | Max (ns) | Avg (ns) | Std Dev |
|-----------|----------|----------|----------|---------|
| __bench_uprobe_uretprobe | 174.10 | 239.56 | 196.79 | 21.31 |
| __bench_uretprobe | 169.87 | 295.02 | 197.06 | 34.42 |
| __bench_uprobe | 179.34 | 429.95 | 233.23 | 69.89 |
| __bench_read | 10508.48 | 13687.01 | 11847.26 | 1083.90 |
| __bench_write | 10259.85 | 14321.91 | 11636.15 | 1196.29 |
| __bench_hash_map_update | 32469.64 | 43609.21 | 37445.01 | 3711.02 |
| __bench_hash_map_lookup | 33597.16 | 43795.69 | 37698.30 | 2983.71 |
| __bench_hash_map_delete | 18632.17 | 24974.57 | 21429.89 | 1906.98 |
| __bench_array_map_update | 13704.65 | 19370.81 | 15577.50 | 1751.59 |
| __bench_array_map_lookup | 12884.14 | 19134.35 | 15329.19 | 2113.11 |
| __bench_array_map_delete | 12723.94 | 16857.29 | 14443.25 | 1447.23 |
| __bench_per_cpu_hash_map_update | 77958.47 | 105199.28 | 89354.41 | 7374.02 |
| __bench_per_cpu_hash_map_lookup | 59281.19 | 73175.65 | 64278.19 | 4737.54 |
| __bench_per_cpu_hash_map_delete | 65039.94 | 90497.46 | 75378.81 | 8447.67 |
| __bench_per_cpu_array_map_update | 23634.05 | 37641.66 | 28365.97 | 3530.88 |
| __bench_per_cpu_array_map_lookup | 17119.52 | 27462.59 | 19804.70 | 3006.58 |
| __bench_per_cpu_array_map_delete | 12634.94 | 15814.25 | 14087.53 | 1088.89 |

### Embedded VM Performance

| Operation | Min (ns) | Max (ns) | Avg (ns) | Std Dev |
|-----------|----------|----------|----------|---------|
| embed | ∞ | ∞ | ∞ | 0.00 |

## Benchmark Metadata

- **Number of runs:** 10
- **Timestamp:** 2025-04-28 16:06:04
- **Total duration:** 705.64 seconds


## Notes

⚠️ The embedded VM benchmark reported infinity values, which indicates failures or timeouts.