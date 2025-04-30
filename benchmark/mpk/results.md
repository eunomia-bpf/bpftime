# BPFtime MPK Benchmark Results

*Generated on 2025-04-30 05:59:31*

## Environment

- **OS:** Linux 6.11.0-24-generic
- **CPU:** Intel(R) Core(TM) Ultra 7 258V (4 cores, 4 threads)
- **Memory:** 15.07 GB
- **Python:** 3.12.7

## Summary

This benchmark compares two different userspace eBPF execution environments:
- **MPK-enabled**: BPFtime with Memory Protection Keys (MPK) enabled
- **Standard**: BPFtime without Memory Protection Keys

*Times shown in nanoseconds (ns) - lower is better*

### Performance Summary

| Operation | MPK (ns) | Standard (ns) | Difference (ns) | Overhead |
|-----------|----------|---------------|-----------------|----------|
| __bench_array_map_delete | 2778.11 | 3792.13 | -1014.02 | -26.74% |
| __bench_array_map_lookup | 3214.49 | 4840.34 | -1625.85 | -33.59% |
| __bench_array_map_update | 4174.81 | 6154.38 | -1979.58 | -32.17% |
| __bench_hash_map_delete | 9271.39 | 13670.12 | -4398.73 | -32.18% |
| __bench_hash_map_lookup | 19804.24 | 29094.03 | -9289.78 | -31.93% |
| __bench_hash_map_update | 20959.01 | 31264.76 | -10305.75 | -32.96% |
| __bench_per_cpu_array_map_delete | 3326.91 | 3847.16 | -520.25 | -13.52% |
| __bench_per_cpu_array_map_lookup | 6630.61 | 7886.82 | -1256.21 | -15.93% |
| __bench_per_cpu_array_map_update | 14043.85 | 18300.46 | -4256.61 | -23.26% |
| __bench_per_cpu_hash_map_delete | 54453.99 | 91674.43 | -37220.44 | -40.60% |
| __bench_per_cpu_hash_map_lookup | 43507.93 | 69107.60 | -25599.67 | -37.04% |
| __bench_per_cpu_hash_map_update | 68188.81 | 104546.51 | -36357.70 | -34.78% |
| __bench_read | 1342.30 | 2581.53 | -1239.23 | -48.00% |
| __bench_uprobe | 164.98 | 341.51 | -176.53 | -51.69% |
| __bench_uprobe_uretprobe | 175.83 | 367.54 | -191.71 | -52.16% |
| __bench_uretprobe | 169.76 | 359.88 | -190.12 | -52.83% |
| __bench_write | 1365.74 | 2167.83 | -802.09 | -37.00% |

### Detailed Comparison

| Operation | Environment | Min (ns) | Max (ns) | Avg (ns) | Std Dev |
|-----------|-------------|----------|----------|----------|---------|
| __bench_array_map_delete | MPK | 2679.80 | 2886.00 | 2778.11 | 84.45 |
| __bench_array_map_delete | Standard | 3743.11 | 3848.11 | 3792.13 | 43.15 |
| __bench_array_map_lookup | MPK | 3042.23 | 3342.11 | 3214.49 | 126.43 |
| __bench_array_map_lookup | Standard | 4531.40 | 5258.96 | 4840.34 | 306.99 |
| __bench_array_map_update | MPK | 3964.33 | 4344.00 | 4174.81 | 157.72 |
| __bench_array_map_update | Standard | 5711.62 | 6829.18 | 6154.38 | 484.85 |
| __bench_hash_map_delete | MPK | 8796.90 | 9785.92 | 9271.39 | 404.76 |
| __bench_hash_map_delete | Standard | 13261.39 | 14253.78 | 13670.12 | 423.60 |
| __bench_hash_map_lookup | MPK | 19370.49 | 20444.82 | 19804.24 | 462.33 |
| __bench_hash_map_lookup | Standard | 28259.11 | 30072.25 | 29094.03 | 747.12 |
| __bench_hash_map_update | MPK | 20541.19 | 21196.57 | 20959.01 | 296.37 |
| __bench_hash_map_update | Standard | 29255.16 | 33612.86 | 31264.76 | 1795.05 |
| __bench_per_cpu_array_map_delete | MPK | 2734.15 | 4410.69 | 3326.91 | 767.47 |
| __bench_per_cpu_array_map_delete | Standard | 3756.45 | 3919.14 | 3847.16 | 67.73 |
| __bench_per_cpu_array_map_lookup | MPK | 5823.84 | 7696.44 | 6630.61 | 786.13 |
| __bench_per_cpu_array_map_lookup | Standard | 7814.66 | 8004.37 | 7886.82 | 83.84 |
| __bench_per_cpu_array_map_update | MPK | 13728.22 | 14256.92 | 14043.85 | 227.70 |
| __bench_per_cpu_array_map_update | Standard | 17645.66 | 19149.41 | 18300.46 | 629.06 |
| __bench_per_cpu_hash_map_delete | MPK | 50084.97 | 62185.52 | 54453.99 | 5482.44 |
| __bench_per_cpu_hash_map_delete | Standard | 76354.61 | 103287.43 | 91674.43 | 11303.38 |
| __bench_per_cpu_hash_map_lookup | MPK | 42879.29 | 44423.93 | 43507.93 | 662.53 |
| __bench_per_cpu_hash_map_lookup | Standard | 67155.98 | 71104.16 | 69107.60 | 1612.15 |
| __bench_per_cpu_hash_map_update | MPK | 66451.53 | 69652.10 | 68188.81 | 1320.91 |
| __bench_per_cpu_hash_map_update | Standard | 99985.87 | 107253.71 | 104546.51 | 3243.64 |
| __bench_read | MPK | 1286.32 | 1411.95 | 1342.30 | 52.19 |
| __bench_read | Standard | 2026.90 | 3650.44 | 2581.53 | 756.01 |
| __bench_uprobe | MPK | 164.07 | 165.81 | 164.98 | 0.71 |
| __bench_uprobe | Standard | 258.83 | 498.64 | 341.51 | 111.16 |
| __bench_uprobe_uretprobe | MPK | 170.31 | 180.57 | 175.83 | 4.22 |
| __bench_uprobe_uretprobe | Standard | 273.33 | 554.36 | 367.54 | 132.10 |
| __bench_uretprobe | MPK | 165.01 | 176.88 | 169.76 | 5.13 |
| __bench_uretprobe | Standard | 262.39 | 528.42 | 359.88 | 119.66 |
| __bench_write | MPK | 1282.43 | 1437.23 | 1365.74 | 63.75 |
| __bench_write | Standard | 1991.73 | 2414.75 | 2167.83 | 179.81 |

## Benchmark Metadata

- **Number of runs:** 3
- **Timestamp:** 2025-04-30 05:59:31
- **Total duration:** 198.56 seconds
