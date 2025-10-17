
# CUDA Benchmark Results

**Device:** NVIDIA GeForce RTX 5090  
**Timestamp:** 2025-10-16T15:56:15.321219  

## Workload Configuration

| Workload | Elements | Iterations | Threads | Blocks |
|----------|----------|------------|---------|--------|
| large | 100000 | 1000 | 512 | 196 |
| medium | 10000 | 10000 | 256 | 40 |
| minimal | 32 | 3 | 32 | 1 |
| small | 1000 | 10000 | 256 | 4 |
| small_64x16 | 1000 | 10000 | 64 | 16 |
| tiny | 32 | 10000 | 32 | 1 |
| xlarge | 1000000 | 1000 | 512 | 1954 |

## Benchmark Results

| Test Name | Workload | Avg Time (μs) | vs Baseline | Overhead |
|-----------|----------|---------------|-------------|----------|
| Baseline (tiny) | tiny | 5.21 | - | - |
| Baseline (small) | small | 5.26 | - | - |
| Baseline (medium) | medium | 5.30 | - | - |
| Baseline (large) | large | 5.55 | - | - |
| Baseline (xlarge) | xlarge | 7.06 | - | - |
| Empty probe (tiny) | tiny | 6.25 | 5.21 | 1.20x (+20.0%) |
| Empty probe (small) | small | 6.35 | 5.26 | 1.21x (+20.7%) |
| Empty probe (medium) | medium | 6.38 | 5.30 | 1.20x (+20.4%) |
| Empty probe (large) | large | 6.70 | 5.55 | 1.21x (+20.7%) |
| Empty probe (xlarge) | xlarge | 10.89 | 7.06 | 1.54x (+54.2%) |
| Entry probe (tiny) | tiny | 6.24 | 5.21 | 1.20x (+19.8%) |
| Entry probe (small) | small | 6.42 | 5.26 | 1.22x (+22.1%) |
| Entry probe (medium) | medium | 6.44 | 5.30 | 1.22x (+21.5%) |
| Entry probe (xlarge) | xlarge | 10.82 | 7.06 | 1.53x (+53.3%) |
| Exit probe (tiny) | tiny | 6.25 | 5.21 | 1.20x (+20.0%) |
| Exit probe (small) | small | 6.27 | 5.26 | 1.19x (+19.2%) |
| Exit probe (large) | large | 6.67 | 5.55 | 1.20x (+20.2%) |
| Entry+Exit (tiny) | tiny | 6.32 | 5.21 | 1.21x (+21.3%) |
| Entry+Exit (small_64x16) | small_64x16 | 6.38 | 5.26 | 1.21x (+21.3%) |
| Entry+Exit (small) | small | 6.36 | 5.26 | 1.21x (+20.9%) |
| Entry+Exit (medium) | medium | 6.35 | 5.30 | 1.20x (+19.8%) |
| Entry+Exit (large) | large | 6.69 | 5.55 | 1.21x (+20.5%) |
| Entry+Exit (xlarge) | xlarge | 10.90 | 7.06 | 1.54x (+54.4%) |
| GPU Ringbuf (tiny) | tiny | 10.58 | 5.21 | 2.03x (+103.1%) |
| GPU Ringbuf (small_64x16) | small_64x16 | 16.18 | 5.26 | 3.08x (+207.6%) |
| GPU Ringbuf (small) | small | 15.99 | 5.26 | 3.04x (+204.0%) |
| GPU Ringbuf (medium) | medium | 33.45 | 5.30 | 6.31x (+531.1%) |
| GPU Ringbuf (large) | large | 1.10 | 5.55 | 0.19x (-80.2%) |
| GPU Ringbuf (xlarge) | xlarge | 1.12 | 7.06 | 0.15x (-84.1%) |
| Global timer (tiny) | tiny | 6.99 | 5.21 | 1.34x (+34.2%) |
| Global timer (small) | small | 7.23 | 5.26 | 1.37x (+37.5%) |
| Global timer (medium) | medium | 7.21 | 5.30 | 1.36x (+36.0%) |
| Global timer (xlarge) | xlarge | 31.60 | 7.06 | 4.48x (+347.6%) |
| Per-GPU-thread array (tiny) | tiny | 9.53 | 5.21 | 1.83x (+82.9%) |
| Per-GPU-thread array (small_64x16) | small_64x16 | 9.77 | 5.26 | 1.86x (+85.7%) |
| Per-GPU-thread array (small) | small | 9.83 | 5.26 | 1.87x (+86.9%) |
| Per-GPU-thread array (medium) | medium | 9.88 | 5.30 | 1.86x (+86.4%) |
| Per-GPU-thread array (large) | large | 18.35 | 5.55 | 3.31x (+230.6%) |
| Per-GPU-thread array (xlarge) | xlarge | 1.07 | 7.06 | 0.15x (-84.8%) |
| Memtrace (tiny) | tiny | 9.41 | 5.21 | 1.81x (+80.6%) |
| Memtrace (small) | small | 9.83 | 5.26 | 1.87x (+86.9%) |
| Memtrace (medium) | medium | 9.82 | 5.30 | 1.85x (+85.3%) |
| Memtrace (large) | large | 23.49 | 5.55 | 4.23x (+323.2%) |
| GPU Array map update (tiny) | tiny | 11.14 | 5.21 | 2.14x (+113.8%) |
| GPU Array map update (small_64x16) | small_64x16 | 11.29 | 5.26 | 2.15x (+114.6%) |
| GPU Array map update (small) | small | 12.73 | 5.26 | 2.42x (+142.0%) |
| GPU Array map update (medium) | medium | 13.37 | 5.30 | 2.52x (+152.3%) |
| GPU Array map update (large) | large | 28.41 | 5.55 | 5.12x (+411.9%) |
| GPU Array map update (xlarge) | xlarge | 144.47 | 7.06 | 20.46x (+1946.3%) |
| GPU Array map lookup (tiny) | tiny | 8.18 | 5.21 | 1.57x (+57.0%) |
| GPU Array map lookup (small_64x16) | small_64x16 | 8.32 | 5.26 | 1.58x (+58.2%) |
| GPU Array map lookup (small) | small | 8.27 | 5.26 | 1.57x (+57.2%) |
| GPU Array map lookup (medium) | medium | 8.33 | 5.30 | 1.57x (+57.2%) |
| GPU Array map lookup (large) | large | 12.49 | 5.55 | 2.25x (+125.0%) |
| GPU Array map lookup (xlarge) | xlarge | 43.62 | 7.06 | 6.18x (+517.8%) |
| CPU Array map update (minimal) | minimal | 33447.65 | 5.21 | 6419.89x (+641889.4%) |
| CPU Array map lookup (minimal) | minimal | 33440.58 | 5.21 | 6418.54x (+641753.7%) |
| CPU Hash map update (minimal) | minimal | 33439.98 | 5.21 | 6418.42x (+641742.2%) |
| CPU Hash map lookup (minimal) | minimal | 33451.69 | 5.21 | 6420.67x (+641967.0%) |
| CPU Hash map delete (minimal) | minimal | 33452.59 | 5.21 | 6420.84x (+641984.3%) |

