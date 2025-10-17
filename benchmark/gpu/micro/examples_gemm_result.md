
# CUDA Benchmark Results

**Device:** NVIDIA GeForce RTX 5090  
**Timestamp:** 2025-10-16T20:28:16.532134  

## Workload Configuration

| Workload | Elements | Iterations | Threads | Blocks |
|----------|----------|------------|---------|--------|
| large | benchmark/gpu/workload/matrixMul | 100 | 100 | 512 |
| medium | benchmark/gpu/workload/matrixMul | 100 | 100 | 256 |
| minimal | benchmark/gpu/workload/matrixMul | 32 | 3 | 32 |
| small | benchmark/gpu/workload/matrixMul | 1000 | 100 | 256 |
| tiny | benchmark/gpu/workload/matrixMul | 32 | 100 | 32 |
| xlarge | benchmark/gpu/workload/matrixMul | 100 | 100 | 512 |

## Benchmark Results

| Test Name | Workload | Avg Time (μs) | vs Baseline | Overhead |
|-----------|----------|---------------|-------------|----------|
| Baseline (minimal) | minimal | 6.78 | - | - |
| Baseline (small) | small | 216.50 | - | - |
| CUDA Counter (minimal) | minimal | 8.57 | 6.78 | 1.26x (+26.4%) |
| Memory Trace (small) | small | 1167.03 | 216.50 | 5.39x (+439.0%) |
| Thread Histogram (small) | small | 343.07 | 216.50 | 1.58x (+58.5%) |
| Launch Latency (minimal) | minimal | 9.70 | 6.78 | 1.43x (+43.1%) |
| Kernel Retsnoop (small) | small | 342.32 | 216.50 | 1.58x (+58.1%) |

