
# CUDA Benchmark Results

**Device:** NVIDIA GeForce RTX 5090  
**Timestamp:** 2025-10-16T20:44:42.720621  

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
| Baseline (tiny) | tiny | 5.92 | - | - |
| Baseline (small) | small | 216.90 | - | - |
| Baseline (medium) | medium | 225407.68 | - | - |
| Baseline (large) | large | 225662.02 | - | - |
| Baseline (xlarge) | xlarge | 225704.96 | - | - |
| Empty probe (tiny) | tiny | 8.15 | 5.92 | 1.38x (+37.7%) |
| Empty probe (small) | small | 345.14 | 216.90 | 1.59x (+59.1%) |
| Empty probe (medium) | medium | 362207.90 | 225407.68 | 1.61x (+60.7%) |
| Empty probe (large) | large | 362312.15 | 225662.02 | 1.61x (+60.6%) |
| Empty probe (xlarge) | xlarge | 362356.45 | 225704.96 | 1.61x (+60.5%) |
| Entry probe (tiny) | tiny | 9.35 | 5.92 | 1.58x (+57.9%) |
| Entry probe (small) | small | 344.41 | 216.90 | 1.59x (+58.8%) |
| Entry probe (medium) | medium | 362223.73 | 225407.68 | 1.61x (+60.7%) |
| Entry probe (xlarge) | xlarge | 362398.57 | 225704.96 | 1.61x (+60.6%) |
| Exit probe (tiny) | tiny | 8.21 | 5.92 | 1.39x (+38.7%) |
| Exit probe (small) | small | 343.89 | 216.90 | 1.59x (+58.5%) |
| Exit probe (large) | large | 362215.59 | 225662.02 | 1.61x (+60.5%) |
| Entry+Exit (tiny) | tiny | 8.21 | 5.92 | 1.39x (+38.7%) |
| Entry+Exit (small_64x16) | small_64x16 | 343.93 | 216.90 | 1.59x (+58.6%) |
| Entry+Exit (small) | small | 344.04 | 216.90 | 1.59x (+58.6%) |
| Entry+Exit (medium) | medium | 362124.47 | 225407.68 | 1.61x (+60.7%) |
| Entry+Exit (large) | large | 362379.75 | 225662.02 | 1.61x (+60.6%) |
| Entry+Exit (xlarge) | xlarge | 362412.03 | 225704.96 | 1.61x (+60.6%) |
| GPU Ringbuf (tiny) | tiny | 8.19 | 5.92 | 1.38x (+38.3%) |
| GPU Ringbuf (small_64x16) | small_64x16 | 344.16 | 216.90 | 1.59x (+58.7%) |
| GPU Ringbuf (small) | small | 344.09 | 216.90 | 1.59x (+58.6%) |
| GPU Ringbuf (medium) | medium | 362094.83 | 225407.68 | 1.61x (+60.6%) |
| GPU Ringbuf (large) | large | 362358.23 | 225662.02 | 1.61x (+60.6%) |
| GPU Ringbuf (xlarge) | xlarge | 362380.52 | 225704.96 | 1.61x (+60.6%) |
| Global timer (tiny) | tiny | 8.22 | 5.92 | 1.39x (+38.9%) |
| Global timer (small) | small | 343.79 | 216.90 | 1.59x (+58.5%) |
| Global timer (medium) | medium | 362252.01 | 225407.68 | 1.61x (+60.7%) |
| Global timer (xlarge) | xlarge | 362360.90 | 225704.96 | 1.61x (+60.5%) |
| Per-GPU-thread array (tiny) | tiny | 8.27 | 5.92 | 1.40x (+39.7%) |
| Per-GPU-thread array (small_64x16) | small_64x16 | 343.96 | 216.90 | 1.59x (+58.6%) |
| Per-GPU-thread array (small) | small | 343.39 | 216.90 | 1.58x (+58.3%) |
| Per-GPU-thread array (medium) | medium | 362147.10 | 225407.68 | 1.61x (+60.7%) |
| Per-GPU-thread array (large) | large | 362420.74 | 225662.02 | 1.61x (+60.6%) |
| Per-GPU-thread array (xlarge) | xlarge | 362399.61 | 225704.96 | 1.61x (+60.6%) |
| Memtrace (tiny) | tiny | 13.65 | 5.92 | 2.31x (+130.6%) |
| Memtrace (small) | small | 556.51 | 216.90 | 2.57x (+156.6%) |
| Memtrace (medium) | medium | 464351.33 | 225407.68 | 2.06x (+106.0%) |
| Memtrace (large) | large | 464446.82 | 225662.02 | 2.06x (+105.8%) |
| GPU Array map update (tiny) | tiny | 8.22 | 5.92 | 1.39x (+38.9%) |
| GPU Array map update (small_64x16) | small_64x16 | 343.90 | 216.90 | 1.59x (+58.6%) |
| GPU Array map update (small) | small | 343.78 | 216.90 | 1.58x (+58.5%) |
| GPU Array map update (medium) | medium | 362168.27 | 225407.68 | 1.61x (+60.7%) |
| GPU Array map update (large) | large | 362374.34 | 225662.02 | 1.61x (+60.6%) |
| GPU Array map update (xlarge) | xlarge | 362422.68 | 225704.96 | 1.61x (+60.6%) |
| GPU Array map lookup (tiny) | tiny | 8.22 | 5.92 | 1.39x (+38.9%) |
| GPU Array map lookup (small_64x16) | small_64x16 | 343.95 | 216.90 | 1.59x (+58.6%) |
| GPU Array map lookup (small) | small | 343.45 | 216.90 | 1.58x (+58.3%) |
| GPU Array map lookup (medium) | medium | 362126.09 | 225407.68 | 1.61x (+60.7%) |
| GPU Array map lookup (large) | large | 362391.42 | 225662.02 | 1.61x (+60.6%) |
| GPU Array map lookup (xlarge) | xlarge | 362360.78 | 225704.96 | 1.61x (+60.5%) |
| CPU Array map update (minimal) | minimal | 9.21 | 5.92 | 1.56x (+55.6%) |
| CPU Array map lookup (minimal) | minimal | 8.81 | 5.92 | 1.49x (+48.8%) |
| CPU Hash map update (minimal) | minimal | 8.86 | 5.92 | 1.50x (+49.7%) |
| CPU Hash map lookup (minimal) | minimal | 8.74 | 5.92 | 1.48x (+47.6%) |
| CPU Hash map delete (minimal) | minimal | 8.76 | 5.92 | 1.48x (+48.0%) |

