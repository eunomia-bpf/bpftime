
# CUDA Benchmark Results

**Device:** NVIDIA GeForce RTX 5090  
**Timestamp:** 2025-10-16T01:58:20.009017  

## Workload Configuration

| Workload | Elements | Iterations | Threads | Blocks |
|----------|----------|------------|---------|--------|
| large | 100000 | 1000 | 512 | 196 |
| medium | 10000 | 10000 | 256 | 40 |
| minimal | 32 | 3 | 32 | 1 |
| small | 1000 | 10000 | 256 | 4 |
| tiny | 32 | 10000 | 32 | 1 |
| xlarge | 1000000 | 1000 | 512 | 1954 |

## Benchmark Results

| Test Name | Workload | Avg Time (μs) | vs Baseline | Overhead |
|-----------|----------|---------------|-------------|----------|
| Baseline (minimal) | minimal | 6.52 | - | - |
| CUDA Counter (minimal) | minimal | 337482.88 | 6.52 | 51761.18x (+5176017.8%) |
| Memory Trace (minimal) | minimal | 134815.58 | 6.52 | 20677.24x (+2067623.6%) |
| Thread Histogram (minimal) | minimal | 10.80 | 6.52 | 1.66x (+65.6%) |
| Launch Latency (minimal) | minimal | 8.41 | 6.52 | 1.29x (+29.0%) |
| Kernel Retsnoop (minimal) | minimal | 7.09 | 6.52 | 1.09x (+8.7%) |

