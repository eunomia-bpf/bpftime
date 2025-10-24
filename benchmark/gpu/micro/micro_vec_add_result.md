
# CUDA Benchmark Results

**Device:** NVIDIA GeForce RTX 5090  
**Timestamp:** 2025-10-16T19:30:20.854584  

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
| Baseline (tiny) | tiny | 5.23 | - | - |
| Baseline (small) | small | 5.29 | - | - |
| Baseline (medium) | medium | 5.31 | - | - |
| Baseline (large) | large | 5.60 | - | - |
| Baseline (xlarge) | xlarge | 7.02 | - | - |
| Empty probe (tiny) | tiny | 6.24 | 5.23 | 1.19x (+19.3%) |
| Empty probe (small) | small | 6.33 | 5.29 | 1.20x (+19.7%) |
| Empty probe (medium) | medium | 6.40 | 5.31 | 1.21x (+20.5%) |
| Empty probe (large) | large | 6.75 | 5.60 | 1.21x (+20.5%) |
| Empty probe (xlarge) | xlarge | 10.87 | 7.02 | 1.55x (+54.8%) |
| Entry probe (tiny) | tiny | 6.21 | 5.23 | 1.19x (+18.7%) |
| Entry probe (small) | small | 6.43 | 5.29 | 1.22x (+21.6%) |
| Entry probe (medium) | medium | 6.41 | 5.31 | 1.21x (+20.7%) |
| Entry probe (xlarge) | xlarge | 10.79 | 7.02 | 1.54x (+53.7%) |
| Exit probe (tiny) | tiny | 6.23 | 5.23 | 1.19x (+19.1%) |
| Exit probe (small) | small | 6.28 | 5.29 | 1.19x (+18.7%) |
| Exit probe (large) | large | 6.66 | 5.60 | 1.19x (+18.9%) |
| Entry+Exit (tiny) | tiny | 6.25 | 5.23 | 1.20x (+19.5%) |
| Entry+Exit (small_64x16) | small_64x16 | 6.38 | 5.29 | 1.21x (+20.6%) |
| Entry+Exit (small) | small | 6.29 | 5.29 | 1.19x (+18.9%) |
| Entry+Exit (medium) | medium | 6.38 | 5.31 | 1.20x (+20.2%) |
| Entry+Exit (large) | large | 6.67 | 5.60 | 1.19x (+19.1%) |
| Entry+Exit (xlarge) | xlarge | 10.93 | 7.02 | 1.56x (+55.7%) |
| GPU Ringbuf (tiny) | tiny | 10.63 | 5.23 | 2.03x (+103.3%) |
| GPU Ringbuf (small_64x16) | small_64x16 | 16.17 | 5.29 | 3.06x (+205.7%) |
| GPU Ringbuf (small) | small | 15.91 | 5.29 | 3.01x (+200.8%) |
| GPU Ringbuf (medium) | medium | 33.32 | 5.31 | 6.27x (+527.5%) |
| GPU Ringbuf (large) | large | 1.10 | 5.60 | 0.19x (-80.4%) |
| GPU Ringbuf (xlarge) | xlarge | 1.11 | 7.02 | 0.15x (-84.2%) |
| Global timer (tiny) | tiny | 6.97 | 5.23 | 1.33x (+33.3%) |
| Global timer (small) | small | 7.22 | 5.29 | 1.36x (+36.5%) |
| Global timer (medium) | medium | 7.20 | 5.31 | 1.36x (+35.6%) |
| Global timer (xlarge) | xlarge | 31.57 | 7.02 | 4.50x (+349.7%) |
| Per-GPU-thread array (tiny) | tiny | 9.51 | 5.23 | 1.82x (+81.8%) |
| Per-GPU-thread array (small_64x16) | small_64x16 | 9.75 | 5.29 | 1.84x (+84.3%) |
| Per-GPU-thread array (small) | small | 9.82 | 5.29 | 1.86x (+85.6%) |
| Per-GPU-thread array (medium) | medium | 10.15 | 5.31 | 1.91x (+91.1%) |
| Per-GPU-thread array (large) | large | 17.94 | 5.60 | 3.20x (+220.4%) |
| Per-GPU-thread array (xlarge) | xlarge | 1.13 | 7.02 | 0.16x (-83.9%) |
| Memtrace (tiny) | tiny | 9.42 | 5.23 | 1.80x (+80.1%) |
| Memtrace (small) | small | 9.79 | 5.29 | 1.85x (+85.1%) |
| Memtrace (medium) | medium | 9.81 | 5.31 | 1.85x (+84.7%) |
| Memtrace (large) | large | 23.47 | 5.60 | 4.19x (+319.1%) |
| GPU Array map update (tiny) | tiny | 11.12 | 5.23 | 2.13x (+112.6%) |
| GPU Array map update (small_64x16) | small_64x16 | 11.28 | 5.29 | 2.13x (+113.2%) |
| GPU Array map update (small) | small | 12.73 | 5.29 | 2.41x (+140.6%) |
| GPU Array map update (medium) | medium | 13.34 | 5.31 | 2.51x (+151.2%) |
| GPU Array map update (large) | large | 28.35 | 5.60 | 5.06x (+406.3%) |
| GPU Array map update (xlarge) | xlarge | 141.25 | 7.02 | 20.12x (+1912.1%) |
| GPU Array map lookup (tiny) | tiny | 8.17 | 5.23 | 1.56x (+56.2%) |
| GPU Array map lookup (small_64x16) | small_64x16 | 8.31 | 5.29 | 1.57x (+57.1%) |
| GPU Array map lookup (small) | small | 8.25 | 5.29 | 1.56x (+56.0%) |
| GPU Array map lookup (medium) | medium | 8.34 | 5.31 | 1.57x (+57.1%) |
| GPU Array map lookup (large) | large | 12.43 | 5.60 | 2.22x (+122.0%) |
| GPU Array map lookup (xlarge) | xlarge | 43.73 | 7.02 | 6.23x (+522.9%) |
| CPU Array map update (minimal) | minimal | 33450.16 | 5.23 | 6395.82x (+639482.4%) |
| CPU Array map lookup (minimal) | minimal | 33444.77 | 5.23 | 6394.79x (+639379.3%) |
| CPU Hash map update (minimal) | minimal | 33441.53 | 5.23 | 6394.17x (+639317.4%) |
| CPU Hash map lookup (minimal) | minimal | 33537.08 | 5.23 | 6412.44x (+641144.4%) |
| CPU Hash map delete (minimal) | minimal | 33448.05 | 5.23 | 6395.42x (+639442.1%) |

