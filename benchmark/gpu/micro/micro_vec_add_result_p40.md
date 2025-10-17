
# CUDA Benchmark Results

**Device:** Tesla P40  
**Timestamp:** 2025-10-18T01:05:32.522168  

## Workload Configuration

| Workload | Elements | Iterations | Threads | Blocks |
|----------|----------|------------|---------|--------|
| large | benchmark/gpu/workload/vec_add | 100000 | 1000 | 512 |
| medium | benchmark/gpu/workload/vec_add | 10000 | 10000 | 256 |
| minimal | benchmark/gpu/workload/vec_add | 32 | 3 | 32 |
| small | benchmark/gpu/workload/vec_add | 1000 | 10000 | 256 |
| small_64x16 | benchmark/gpu/workload/vec_add | 1000 | 10000 | 64 |
| tiny | benchmark/gpu/workload/vec_add | 32 | 10000 | 32 |
| xlarge | benchmark/gpu/workload/vec_add | 1000000 | 1000 | 512 |

## Benchmark Results

| Test Name | Workload | Avg Time (μs) | vs Baseline | Overhead |
|-----------|----------|---------------|-------------|----------|
| Baseline (tiny) | tiny | 14.10 | - | - |
| Baseline (small) | small | 16.78 | - | - |
| Baseline (medium) | medium | 14.19 | - | - |
| Baseline (large) | large | 49.12 | - | - |
| Baseline (xlarge) | xlarge | 85.80 | - | - |
| Empty probe (tiny) | tiny | 23.57 | 14.10 | 1.67x (+67.2%) |
| Empty probe (small) | small | 17.52 | 16.78 | 1.04x (+4.4%) |
| Empty probe (medium) | medium | 21.58 | 14.19 | 1.52x (+52.1%) |
| Empty probe (large) | large | 27.80 | 49.12 | 0.56x (-43.4%) |
| Empty probe (xlarge) | xlarge | 127.38 | 85.80 | 1.48x (+48.5%) |
| Entry probe (tiny) | tiny | 17.08 | 14.10 | 1.21x (+21.1%) |
| Entry probe (small) | small | 17.36 | 16.78 | 1.03x (+3.5%) |
| Entry probe (medium) | medium | 24.46 | 14.19 | 1.72x (+72.4%) |
| Entry probe (xlarge) | xlarge | 86.60 | 85.80 | 1.01x (+0.9%) |
| Exit probe (tiny) | tiny | 23.62 | 14.10 | 1.68x (+67.5%) |
| Exit probe (small) | small | 18.49 | 16.78 | 1.10x (+10.2%) |
| Exit probe (large) | large | 59.72 | 49.12 | 1.22x (+21.6%) |
| Entry+Exit (tiny) | tiny | 17.23 | 14.10 | 1.22x (+22.2%) |
| Entry+Exit (small_64x16) | small_64x16 | 20.45 | 16.78 | 1.22x (+21.9%) |
| Entry+Exit (small) | small | 20.71 | 16.78 | 1.23x (+23.4%) |
| Entry+Exit (medium) | medium | 20.69 | 14.19 | 1.46x (+45.8%) |
| Entry+Exit (large) | large | 61.78 | 49.12 | 1.26x (+25.8%) |
| Entry+Exit (xlarge) | xlarge | 120.94 | 85.80 | 1.41x (+41.0%) |
| GPU Ringbuf (tiny) | tiny | 34.64 | 14.10 | 2.46x (+145.7%) |
| GPU Ringbuf (small_64x16) | small_64x16 | 49.72 | 16.78 | 2.96x (+196.3%) |
| GPU Ringbuf (small) | small | 49.37 | 16.78 | 2.94x (+194.2%) |
| GPU Ringbuf (medium) | medium | 142.33 | 14.19 | 10.03x (+903.0%) |
| GPU Ringbuf (large) | large | 2.71 | 49.12 | 0.5x (-94.5%) |
| GPU Ringbuf (xlarge) | xlarge | 2.42 | 85.80 | 0.2x (-97.2%) |
| Global timer (tiny) | tiny | 17.65 | 14.10 | 1.25x (+25.2%) |
| Global timer (small) | small | 18.45 | 16.78 | 1.10x (+10.0%) |
| Global timer (medium) | medium | 18.47 | 14.19 | 1.30x (+30.2%) |
| Global timer (xlarge) | xlarge | 121.96 | 85.80 | 1.42x (+42.1%) |
| Per-GPU-thread array (tiny) | tiny | 30.89 | 14.10 | 2.19x (+119.1%) |
| Per-GPU-thread array (small_64x16) | small_64x16 | 36.02 | 16.78 | 2.15x (+114.7%) |
| Per-GPU-thread array (small) | small | 32.50 | 16.78 | 1.94x (+93.7%) |
| Per-GPU-thread array (medium) | medium | 38.92 | 14.19 | 2.74x (+174.3%) |
| Per-GPU-thread array (large) | large | 151.39 | 49.12 | 3.08x (+208.2%) |
| Per-GPU-thread array (xlarge) | xlarge | 34.47 | 85.80 | 0.40x (-59.8%) |
| Memtrace (tiny) | tiny | 28.54 | 14.10 | 2.02x (+102.4%) |
| Memtrace (small) | small | 27.14 | 16.78 | 1.62x (+61.7%) |
| Memtrace (medium) | medium | 35.50 | 14.19 | 2.50x (+150.2%) |
| Memtrace (large) | large | 285.58 | 49.12 | 5.81x (+481.4%) |
| GPU Array map update (tiny) | tiny | 30.59 | 14.10 | 2.17x (+117.0%) |
| GPU Array map update (small_64x16) | small_64x16 | 31.65 | 16.78 | 1.89x (+88.6%) |
| GPU Array map update (small) | small | 37.10 | 16.78 | 2.21x (+121.1%) |
| GPU Array map update (medium) | medium | 48.46 | 14.19 | 3.42x (+241.5%) |
| GPU Array map update (large) | large | 232.86 | 49.12 | 4.74x (+374.1%) |
| GPU Array map update (xlarge) | xlarge | 2036.86 | 85.80 | 23.74x (+2274.0%) |
| GPU Array map lookup (tiny) | tiny | 27.71 | 14.10 | 1.97x (+96.5%) |
| GPU Array map lookup (small_64x16) | small_64x16 | 26.31 | 16.78 | 1.57x (+56.8%) |
| GPU Array map lookup (small) | small | 23.62 | 16.78 | 1.41x (+40.8%) |
| GPU Array map lookup (medium) | medium | 25.49 | 14.19 | 1.80x (+79.6%) |
| GPU Array map lookup (large) | large | 69.57 | 49.12 | 1.42x (+41.6%) |
| GPU Array map lookup (xlarge) | xlarge | 465.37 | 85.80 | 5.42x (+442.4%) |
| CPU Array map update (minimal) | minimal | 35798.30 | 14.10 | 2538.89x (+253788.7%) |
| CPU Array map lookup (minimal) | minimal | 35975.81 | 14.10 | 2551.48x (+255047.6%) |
| CPU Hash map update (minimal) | minimal | 36022.03 | 14.10 | 2554.75x (+255375.4%) |
| CPU Hash map lookup (minimal) | minimal | 35880.65 | 14.10 | 2544.73x (+254372.7%) |
| CPU Hash map delete (minimal) | minimal | 36062.66 | 14.10 | 2557.64x (+255663.5%) |

