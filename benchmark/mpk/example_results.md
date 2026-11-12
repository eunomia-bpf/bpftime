# BPFtime MPK Benchmark Results

*Generated on 2025-04-30 05:30:30*

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
| __bench_array_map_delete | 2896.06 | 3188.04 | -291.98 | -9.16% |
| __bench_array_map_lookup | 3220.42 | 3617.90 | -397.48 | -10.99% |
| __bench_array_map_update | 4284.50 | 4511.89 | -227.40 | -5.04% |
| __bench_hash_map_delete | 9583.83 | 10629.93 | -1046.10 | -9.84% |
| __bench_hash_map_lookup | 20756.68 | 22439.64 | -1682.96 | -7.50% |
| __bench_hash_map_update | 21778.36 | 24894.10 | -3115.74 | -12.52% |
| __bench_per_cpu_array_map_delete | 2858.53 | 3386.52 | -527.99 | -15.59% |
| __bench_per_cpu_array_map_lookup | 6239.73 | 7603.60 | -1363.87 | -17.94% |
| __bench_per_cpu_array_map_update | 14553.99 | 16098.16 | -1544.17 | -9.59% |
| __bench_per_cpu_hash_map_delete | 59352.15 | 64672.69 | -5320.54 | -8.23% |
| __bench_per_cpu_hash_map_lookup | 47188.17 | 50731.39 | -3543.23 | -6.98% |
| __bench_per_cpu_hash_map_update | 70804.04 | 79450.01 | -8645.96 | -10.88% |
| __bench_read | 1458.36 | 1632.59 | -174.23 | -10.67% |
| __bench_uprobe | 171.88 | 181.63 | -9.75 | -5.37% |
| __bench_uprobe_uretprobe | 215.42 | 202.22 | +13.20 | +6.53% |
| __bench_uretprobe | 175.33 | 198.16 | -22.83 | -11.52% |
| __bench_write | 1552.12 | 1539.66 | +12.46 | +0.81% |

### Detailed Comparison

| Operation | Environment | Min (ns) | Max (ns) | Avg (ns) | Std Dev |
|-----------|-------------|----------|----------|----------|---------|
| __bench_array_map_delete | MPK | 2507.36 | 3230.05 | 2896.06 | 184.59 |
| __bench_array_map_delete | Standard | 2934.84 | 3928.13 | 3188.04 | 291.55 |
| __bench_array_map_lookup | MPK | 2977.25 | 3441.36 | 3220.42 | 117.64 |
| __bench_array_map_lookup | Standard | 3257.22 | 4041.01 | 3617.90 | 219.91 |
| __bench_array_map_update | MPK | 3863.87 | 4637.04 | 4284.50 | 231.14 |
| __bench_array_map_update | Standard | 4243.03 | 4888.03 | 4511.89 | 209.72 |
| __bench_hash_map_delete | MPK | 9200.61 | 10795.71 | 9583.83 | 442.82 |
| __bench_hash_map_delete | Standard | 9923.37 | 11673.35 | 10629.93 | 662.08 |
| __bench_hash_map_lookup | MPK | 19358.79 | 23812.39 | 20756.68 | 1392.45 |
| __bench_hash_map_lookup | Standard | 20344.98 | 25377.34 | 22439.64 | 1619.32 |
| __bench_hash_map_update | MPK | 19765.54 | 23957.97 | 21778.36 | 1314.01 |
| __bench_hash_map_update | Standard | 21658.69 | 28849.57 | 24894.10 | 2361.26 |
| __bench_per_cpu_array_map_delete | MPK | 2616.00 | 3035.16 | 2858.53 | 122.94 |
| __bench_per_cpu_array_map_delete | Standard | 2810.34 | 4994.05 | 3386.52 | 634.59 |
| __bench_per_cpu_array_map_lookup | MPK | 5634.72 | 6750.12 | 6239.73 | 336.51 |
| __bench_per_cpu_array_map_lookup | Standard | 6031.13 | 13130.67 | 7603.60 | 1923.01 |
| __bench_per_cpu_array_map_update | MPK | 13443.82 | 15920.98 | 14553.99 | 743.65 |
| __bench_per_cpu_array_map_update | Standard | 13858.46 | 18060.45 | 16098.16 | 1098.83 |
| __bench_per_cpu_hash_map_delete | MPK | 50997.66 | 71418.06 | 59352.15 | 6357.54 |
| __bench_per_cpu_hash_map_delete | Standard | 57188.63 | 72163.86 | 64672.69 | 5118.20 |
| __bench_per_cpu_hash_map_lookup | MPK | 42930.87 | 56394.91 | 47188.17 | 3661.12 |
| __bench_per_cpu_hash_map_lookup | Standard | 47243.19 | 57319.40 | 50731.39 | 3038.28 |
| __bench_per_cpu_hash_map_update | MPK | 66472.07 | 77693.56 | 70804.04 | 3105.51 |
| __bench_per_cpu_hash_map_update | Standard | 70514.58 | 89767.07 | 79450.01 | 6610.00 |
| __bench_read | MPK | 1363.89 | 1555.76 | 1458.36 | 58.62 |
| __bench_read | Standard | 1483.89 | 1845.51 | 1632.59 | 117.25 |
| __bench_uprobe | MPK | 163.86 | 190.21 | 171.88 | 7.75 |
| __bench_uprobe | Standard | 163.41 | 212.10 | 181.63 | 14.77 |
| __bench_uprobe_uretprobe | MPK | 170.95 | 456.14 | 215.42 | 81.22 |
| __bench_uprobe_uretprobe | Standard | 173.94 | 229.47 | 202.22 | 17.69 |
| __bench_uretprobe | MPK | 161.80 | 192.18 | 175.33 | 8.43 |
| __bench_uretprobe | Standard | 163.25 | 244.01 | 198.16 | 23.24 |
| __bench_write | MPK | 1396.50 | 1764.04 | 1552.12 | 128.29 |
| __bench_write | Standard | 1406.24 | 1709.65 | 1539.66 | 110.58 |

## Benchmark Metadata

- **Number of runs:** 10
- **Timestamp:** 2025-04-30 05:30:30
- **Total duration:** 570.11 seconds
