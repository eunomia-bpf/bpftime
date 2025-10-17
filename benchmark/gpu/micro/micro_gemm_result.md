
# CUDA Benchmark Results

**Device:** NVIDIA GeForce RTX 5090  
**Timestamp:** 2025-10-16T20:28:45.419197  

## Workload Configuration

| Workload | Elements | Iterations | Threads | Blocks |
|----------|----------|------------|---------|--------|
| large | benchmark/gpu/workload/matrixMul | 100000 | 100 | 512 |
| medium | benchmark/gpu/workload/matrixMul | 10000 | 100 | 256 |
| minimal | benchmark/gpu/workload/matrixMul | 32 | 3 | 32 |
| small | benchmark/gpu/workload/matrixMul | 1000 | 100 | 256 |
| small_64x16 | benchmark/gpu/workload/matrixMul | 1000 | 100 | 64 |
| tiny | benchmark/gpu/workload/matrixMul | 32 | 100 | 32 |
| xlarge | benchmark/gpu/workload/matrixMul | 1000000 | 100 | 512 |

## Benchmark Results

| Test Name | Workload | Avg Time (μs) | vs Baseline | Overhead |
|-----------|----------|---------------|-------------|----------|
| Baseline (tiny) | tiny | 5.90 | - | - |
| Baseline (small) | small | 217.58 | - | - |
| Baseline (medium) | medium | 225592.38 | - | - |
| Baseline (large) | large | 1.01 | - | - |
| Baseline (xlarge) | xlarge | FAILED | - | - |
| Empty probe (tiny) | tiny | 8.47 | 5.90 | 1.44x (+43.6%) |
| Empty probe (small) | small | 343.79 | 217.58 | 1.58x (+58.0%) |
| Empty probe (medium) | medium | 364407.97 | 225592.38 | 1.62x (+61.5%) |
| Empty probe (large) | large | 1.29 | 1.01 | 1.28x (+27.7%) |
| Empty probe (xlarge) | xlarge | FAILED | - | - |
| Entry probe (tiny) | tiny | 8.22 | 5.90 | 1.39x (+39.3%) |
| Entry probe (small) | small | 343.71 | 217.58 | 1.58x (+58.0%) |
| Entry probe (medium) | medium | 364381.82 | 225592.38 | 1.62x (+61.5%) |
| Entry probe (xlarge) | xlarge | FAILED | - | - |
| Exit probe (tiny) | tiny | 8.22 | 5.90 | 1.39x (+39.3%) |
| Exit probe (small) | small | 343.11 | 217.58 | 1.58x (+57.7%) |
| Exit probe (large) | large | 1.21 | 1.01 | 1.20x (+19.8%) |
| Entry+Exit (tiny) | tiny | 8.23 | 5.90 | 1.39x (+39.5%) |
| Entry+Exit (small_64x16) | small_64x16 | 343.82 | 217.58 | 1.58x (+58.0%) |
| Entry+Exit (small) | small | 343.55 | 217.58 | 1.58x (+57.9%) |
| Entry+Exit (medium) | medium | 364294.43 | 225592.38 | 1.61x (+61.5%) |
| Entry+Exit (large) | large | 1.35 | 1.01 | 1.34x (+33.7%) |
| Entry+Exit (xlarge) | xlarge | FAILED | - | - |
| GPU Ringbuf (tiny) | tiny | 8.25 | 5.90 | 1.40x (+39.8%) |
| GPU Ringbuf (small_64x16) | small_64x16 | 343.67 | 217.58 | 1.58x (+58.0%) |
| GPU Ringbuf (small) | small | 343.49 | 217.58 | 1.58x (+57.9%) |
| GPU Ringbuf (medium) | medium | 364231.64 | 225592.38 | 1.61x (+61.5%) |
| GPU Ringbuf (large) | large | 1.28 | 1.01 | 1.27x (+26.7%) |
| GPU Ringbuf (xlarge) | xlarge | FAILED | - | - |
| Global timer (tiny) | tiny | 8.21 | 5.90 | 1.39x (+39.2%) |
| Global timer (small) | small | 342.97 | 217.58 | 1.58x (+57.6%) |
| Global timer (medium) | medium | 364375.43 | 225592.38 | 1.62x (+61.5%) |
| Global timer (xlarge) | xlarge | FAILED | - | - |
| Per-GPU-thread array (tiny) | tiny | 8.33 | 5.90 | 1.41x (+41.2%) |
| Per-GPU-thread array (small_64x16) | small_64x16 | 343.47 | 217.58 | 1.58x (+57.9%) |
| Per-GPU-thread array (small) | small | 343.78 | 217.58 | 1.58x (+58.0%) |
| Per-GPU-thread array (medium) | medium | 364602.52 | 225592.38 | 1.62x (+61.6%) |
| Per-GPU-thread array (large) | large | 1.31 | 1.01 | 1.30x (+29.7%) |
| Per-GPU-thread array (xlarge) | xlarge | FAILED | - | - |
| Memtrace (tiny) | tiny | 13.68 | 5.90 | 2.32x (+131.9%) |
| Memtrace (small) | small | 555.00 | 217.58 | 2.55x (+155.1%) |
| Memtrace (medium) | medium | 463793.52 | 225592.38 | 2.06x (+105.6%) |
| Memtrace (large) | large | 1.28 | 1.01 | 1.27x (+26.7%) |
| GPU Array map update (tiny) | tiny | 8.24 | 5.90 | 1.40x (+39.7%) |
| GPU Array map update (small_64x16) | small_64x16 | 343.00 | 217.58 | 1.58x (+57.6%) |
| GPU Array map update (small) | small | 343.54 | 217.58 | 1.58x (+57.9%) |
| GPU Array map update (medium) | medium | 364572.66 | 225592.38 | 1.62x (+61.6%) |
| GPU Array map update (large) | large | 1.30 | 1.01 | 1.29x (+28.7%) |
| GPU Array map update (xlarge) | xlarge | FAILED | - | - |
| GPU Array map lookup (tiny) | tiny | 8.43 | 5.90 | 1.43x (+42.9%) |
| GPU Array map lookup (small_64x16) | small_64x16 | 344.87 | 217.58 | 1.59x (+58.5%) |
| GPU Array map lookup (small) | small | 343.27 | 217.58 | 1.58x (+57.8%) |
| GPU Array map lookup (medium) | medium | 364421.48 | 225592.38 | 1.62x (+61.5%) |
| GPU Array map lookup (large) | large | 1.25 | 1.01 | 1.24x (+23.8%) |
| GPU Array map lookup (xlarge) | xlarge | FAILED | - | - |
| CPU Array map update (minimal) | minimal | 9.30 | 5.90 | 1.58x (+57.6%) |
| CPU Array map lookup (minimal) | minimal | 8.86 | 5.90 | 1.50x (+50.2%) |
| CPU Hash map update (minimal) | minimal | 8.88 | 5.90 | 1.51x (+50.5%) |
| CPU Hash map lookup (minimal) | minimal | 9.02 | 5.90 | 1.53x (+52.9%) |
| CPU Hash map delete (minimal) | minimal | 8.83 | 5.90 | 1.50x (+49.7%) |

