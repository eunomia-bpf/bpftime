
# CUDA Benchmark Results

**Device:** NVIDIA GeForce RTX 5090  
**Timestamp:** 2025-10-16T21:07:06.555716  

## Workload Configuration

| Workload | Elements | Iterations | Threads | Blocks |
|----------|----------|------------|---------|--------|
| large | benchmark/gpu/workload/matrixMul | 10000 | 100 | 512 |
| medium | benchmark/gpu/workload/matrixMul | 10000 | 100 | 256 |
| minimal | benchmark/gpu/workload/matrixMul | 32 | 3 | 32 |
| small | benchmark/gpu/workload/matrixMul | 1000 | 100 | 256 |
| small_64x16 | benchmark/gpu/workload/matrixMul | 1000 | 100 | 64 |
| tiny | benchmark/gpu/workload/matrixMul | 32 | 100 | 32 |
| xlarge | benchmark/gpu/workload/matrixMul | 10000 | 100 | 512 |

## Benchmark Results

| Test Name | Workload | Avg Time (μs) | vs Baseline | Overhead |
|-----------|----------|---------------|-------------|----------|
| Baseline (tiny) | tiny | 5.88 | - | - |
| Baseline (small) | small | 216.60 | - | - |
| Baseline (medium) | medium | 225410.01 | - | - |
| Baseline (large) | large | 225701.83 | - | - |
| Baseline (minimal) | minimal | 6.51 | - | - |
| CUDA Counter (minimal) | minimal | 9.10 | 6.51 | 1.40x (+39.8%) |
| Memory Trace (small) | small | 1171.75 | 216.60 | 5.41x (+441.0%) |
| Thread Histogram (small) | small | 343.21 | 216.60 | 1.58x (+58.5%) |
| Launch Latency (minimal) | minimal | 8.58 | 6.51 | 1.32x (+31.8%) |
| Kernel Retsnoop (small) | small | 342.61 | 216.60 | 1.58x (+58.2%) |

