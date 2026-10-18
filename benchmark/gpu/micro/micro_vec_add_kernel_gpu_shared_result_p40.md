
# CUDA Benchmark Results

**Device:** Tesla P40  
**Timestamp:** 2025-11-30T22:13:32.759155  

## Workload Configuration

| Workload | Binary | Elements | Iterations | Threads | Blocks |
|----------|--------|----------|------------|---------|--------|
| large | benchmark/gpu/workload/vec_add | 100000 | 1000 | 512 | 196 |
| medium | benchmark/gpu/workload/vec_add | 10000 | 10000 | 256 | 40 |
| minimal | benchmark/gpu/workload/vec_add | 32 | 3 | 32 | 1 |
| small | benchmark/gpu/workload/vec_add | 1000 | 10000 | 256 | 4 |
| small_64x16 | benchmark/gpu/workload/vec_add | 1000 | 10000 | 64 | 16 |
| tiny | benchmark/gpu/workload/vec_add | 32 | 10000 | 32 | 1 |
| xlarge | benchmark/gpu/workload/vec_add | 1000000 | 1000 | 512 | 1954 |

## Benchmark Results

| Test Name | Workload | Avg Time (μs) | vs Baseline | Overhead |
|-----------|----------|---------------|-------------|----------|
| Baseline (tiny) | tiny | 7.73 | - | - |
| Baseline (small) | small | 7.77 | - | - |
| Baseline (medium) | medium | 7.93 | - | - |
| Baseline (large) | large | 9.97 | - | - |
| Baseline (xlarge) | xlarge | 55.54 | - | - |
| Kernel-GPU Array map update (tiny) | tiny | 15.58 | 7.73 | 2.02x (+101.6%) |
| Kernel-GPU Array map update (small_64x16) | small_64x16 | 47.69 | 7.77 | 6.14x (+513.8%) |
| Kernel-GPU Array map update (small) | small | 46.75 | 7.77 | 6.02x (+501.7%) |
| Kernel-GPU Array map update (medium) | medium | 356.84 | 7.93 | 45.00x (+4399.9%) |
| Kernel-GPU Array map update (large) | large | 3444.50 | 9.97 | 345.49x (+34448.6%) |
| Kernel-GPU Array map update (xlarge) | xlarge | 34035.91 | 55.54 | 612.82x (+61181.8%) |
| Kernel-GPU Array map lookup (tiny) | tiny | 11.10 | 7.73 | 1.44x (+43.6%) |
| Kernel-GPU Array map lookup (small_64x16) | small_64x16 | 11.03 | 7.77 | 1.42x (+42.0%) |
| Kernel-GPU Array map lookup (small) | small | 12.53 | 7.77 | 1.61x (+61.3%) |
| Kernel-GPU Array map lookup (medium) | medium | 13.48 | 7.93 | 1.70x (+70.0%) |
| Kernel-GPU Array map lookup (large) | large | 53.42 | 9.97 | 5.36x (+435.8%) |
| Kernel-GPU Array map lookup (xlarge) | xlarge | 465.10 | 55.54 | 8.37x (+737.4%) |

