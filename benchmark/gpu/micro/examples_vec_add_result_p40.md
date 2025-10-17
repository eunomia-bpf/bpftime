
# CUDA Benchmark Results

**Device:** Tesla P40  
**Timestamp:** 2025-10-18T01:26:46.480040  

## Workload Configuration

| Workload | Binary | Elements | Iterations | Threads | Blocks |
|----------|--------|----------|------------|---------|--------|
| large | benchmark/gpu/workload/vec_add | 100000 | 1000 | 512 | 196 |
| medium | benchmark/gpu/workload/vec_add | 10000 | 10000 | 256 | 40 |
| minimal | benchmark/gpu/workload/vec_add | 32 | 3 | 32 | 1 |
| small | benchmark/gpu/workload/vec_add | 1000 | 10000 | 256 | 4 |
| tiny | benchmark/gpu/workload/vec_add | 32 | 10000 | 32 | 1 |
| xlarge | benchmark/gpu/workload/vec_add | 1000000 | 1000 | 512 | 1954 |

## Benchmark Results

| Test Name | Workload | Avg Time (μs) | vs Baseline | Overhead |
|-----------|----------|---------------|-------------|----------|
| Baseline (minimal) | minimal | 14.97 | - | - |
| CUDA Counter (minimal) | minimal | 370792.91 | 14.97 | 24769.07x (+2476806.5%) |
| Memory Trace (minimal) | minimal | 60.03 | 14.97 | 4.01x (+301.0%) |
| Thread Histogram (minimal) | minimal | 32.30 | 14.97 | 2.16x (+115.8%) |
| Launch Latency (minimal) | minimal | 20.66 | 14.97 | 1.38x (+38.0%) |
| Kernel Retsnoop (minimal) | minimal | 28.42 | 14.97 | 1.90x (+89.8%) |

