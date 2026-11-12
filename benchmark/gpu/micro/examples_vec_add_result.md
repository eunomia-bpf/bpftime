
# CUDA Benchmark Results

**Device:** NVIDIA GeForce RTX 5090  
**Timestamp:** 2025-10-16T19:28:25.566336  

## Workload Configuration

| Workload | Elements | Iterations | Threads | Blocks |
|----------|----------|------------|---------|--------|
| large | benchmark/gpu/workload/vec_add | 100000 | 1000 | 512 |
| medium | benchmark/gpu/workload/vec_add | 10000 | 10000 | 256 |
| minimal | benchmark/gpu/workload/vec_add | 32 | 3 | 32 |
| small | benchmark/gpu/workload/vec_add | 1000 | 10000 | 256 |
| tiny | benchmark/gpu/workload/vec_add | 32 | 10000 | 32 |
| xlarge | benchmark/gpu/workload/vec_add | 1000000 | 1000 | 512 |

## Benchmark Results

| Test Name | Workload | Avg Time (μs) | vs Baseline | Overhead |
|-----------|----------|---------------|-------------|----------|
| Baseline (minimal) | minimal | 6.47 | - | - |
| CUDA Counter (minimal) | minimal | 337453.27 | 6.47 | 52156.61x (+5215561.1%) |
| Memory Trace (minimal) | minimal | 20.11 | 6.47 | 3.11x (+210.8%) |
| Thread Histogram (minimal) | minimal | 10.82 | 6.47 | 1.67x (+67.2%) |
| Launch Latency (minimal) | minimal | 7.62 | 6.47 | 1.18x (+17.8%) |
| Kernel Retsnoop (minimal) | minimal | 7.25 | 6.47 | 1.12x (+12.1%) |

