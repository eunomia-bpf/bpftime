
# CUDA Benchmark Results

**Device:** NVIDIA GeForce RTX 5090  
**Timestamp:** 2025-10-16T01:22:39.063760  

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
| CUDA Counter (minimal) | minimal | 337996.00 | 6.52 | 51839.88x (+5183887.7%) |
| Memory Trace (minimal) | minimal | 134768.51 | 6.52 | 20670.02x (+2066901.7%) |
| Thread Histogram (minimal) | minimal | 10.94 | 6.52 | 1.68x (+67.8%) |
| Launch Latency (minimal) | minimal | 8.42 | 6.52 | 1.29x (+29.1%) |
| Kernel Retsnoop (minimal) | minimal | 8.77 | 6.52 | 1.35x (+34.5%) |
