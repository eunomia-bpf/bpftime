
# CUDA Benchmark Results

**Device:** NVIDIA GeForce RTX 5090  
**Timestamp:** 2025-10-16T15:55:42.775739  

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
| Baseline (minimal) | minimal | 6.51 | - | - |
| CUDA Counter (minimal) | minimal | 337347.72 | 6.51 | 51819.93x (+5181892.6%) |
| Memory Trace (minimal) | minimal | 20.05 | 6.51 | 3.08x (+208.0%) |
| Thread Histogram (minimal) | minimal | 10.88 | 6.51 | 1.67x (+67.1%) |
| Launch Latency (minimal) | minimal | 7.20 | 6.51 | 1.11x (+10.6%) |
| Kernel Retsnoop (minimal) | minimal | 7.24 | 6.51 | 1.11x (+11.2%) |

