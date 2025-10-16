
# CUDA Benchmark Results

**Device:** NVIDIA GeForce RTX 5090  
**Timestamp:** 2025-10-16T00:40:01.660403  

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
| Baseline (tiny) | tiny | 5.19 | - | - |
| Baseline (small) | small | 5.26 | - | - |
| Baseline (medium) | medium | 5.32 | - | - |
| Baseline (large) | large | 5.57 | - | - |
| Baseline (xlarge) | xlarge | 7.07 | - | - |
| Empty probe (tiny) | tiny | 6.25 | 5.19 | 1.20x (+20.4%) |
| Empty probe (small) | small | 6.28 | 5.26 | 1.19x (+19.4%) |
| Empty probe (medium) | medium | 6.34 | 5.32 | 1.19x (+19.2%) |
| Empty probe (large) | large | 6.68 | 5.57 | 1.20x (+19.9%) |
| Empty probe (xlarge) | xlarge | 10.83 | 7.07 | 1.53x (+53.2%) |
| Entry probe (tiny) | tiny | 6.15 | 5.19 | 1.18x (+18.5%) |
| Entry probe (small) | small | 6.33 | 5.26 | 1.20x (+20.3%) |
| Entry probe (medium) | medium | 6.35 | 5.32 | 1.19x (+19.4%) |
| Entry probe (xlarge) | xlarge | 10.79 | 7.07 | 1.53x (+52.6%) |
| Exit probe (tiny) | tiny | 6.20 | 5.19 | 1.19x (+19.5%) |
| Exit probe (small) | small | 6.24 | 5.26 | 1.19x (+18.6%) |
| Exit probe (large) | large | 6.62 | 5.57 | 1.19x (+18.9%) |
| Entry+Exit (tiny) | tiny | 6.24 | 5.19 | 1.20x (+20.2%) |
| Entry+Exit (small_64x16) | small_64x16 | 6.30 | 5.26 | 1.20x (+19.8%) |
| Entry+Exit (small) | small | 6.25 | 5.26 | 1.19x (+18.8%) |
| Entry+Exit (medium) | medium | 6.35 | 5.32 | 1.19x (+19.4%) |
| Entry+Exit (large) | large | 6.66 | 5.57 | 1.20x (+19.6%) |
| Entry+Exit (xlarge) | xlarge | 10.83 | 7.07 | 1.53x (+53.2%) |
| GPU Ringbuf (tiny) | tiny | 10.51 | 5.19 | 2.03x (+102.5%) |
| GPU Ringbuf (small_64x16) | small_64x16 | 16.09 | 5.26 | 3.06x (+205.9%) |
| GPU Ringbuf (small) | small | 15.96 | 5.26 | 3.03x (+203.4%) |
| GPU Ringbuf (medium) | medium | 33.25 | 5.32 | 6.25x (+525.0%) |
| GPU Ringbuf (large) | large | 1.10 | 5.57 | 0.19x (-80.3%) |
| GPU Ringbuf (xlarge) | xlarge | 1.07 | 7.07 | 0.15x (-84.9%) |
| Global timer (tiny) | tiny | 6.87 | 5.19 | 1.32x (+32.4%) |
| Global timer (small) | small | 7.14 | 5.26 | 1.36x (+35.7%) |
| Global timer (medium) | medium | 7.16 | 5.32 | 1.35x (+34.6%) |
| Global timer (xlarge) | xlarge | 31.60 | 7.07 | 4.47x (+347.0%) |
| Per-GPU-thread array (tiny) | tiny | 9.46 | 5.19 | 1.82x (+82.3%) |
| Per-GPU-thread array (small_64x16) | small_64x16 | 9.74 | 5.26 | 1.85x (+85.2%) |
| Per-GPU-thread array (small) | small | 9.74 | 5.26 | 1.85x (+85.2%) |
| Per-GPU-thread array (medium) | medium | 9.84 | 5.32 | 1.85x (+85.0%) |
| Per-GPU-thread array (large) | large | 18.42 | 5.57 | 3.31x (+230.7%) |
| Per-GPU-thread array (xlarge) | xlarge | 1.09 | 7.07 | 0.15x (-84.6%) |
| Memtrace (tiny) | tiny | 9.29 | 5.19 | 1.79x (+79.0%) |
| Memtrace (small) | small | 9.70 | 5.26 | 1.84x (+84.4%) |
| Memtrace (medium) | medium | 9.79 | 5.32 | 1.84x (+84.0%) |
| Memtrace (large) | large | 23.36 | 5.57 | 4.19x (+319.4%) |
| CPU Array map update (minimal) | minimal | 33447.10 | 5.19 | 6444.53x (+644352.8%) |
| CPU Array map lookup (minimal) | minimal | 33478.00 | 5.19 | 6450.48x (+644948.2%) |
| CPU Hash map update (minimal) | minimal | 33755.68 | 5.19 | 6503.98x (+650298.5%) |
| CPU Hash map lookup (minimal) | minimal | 33454.19 | 5.19 | 6445.89x (+644489.4%) |
| CPU Hash map delete (minimal) | minimal | 33443.88 | 5.19 | 6443.91x (+644290.8%) |

## Test Case Descriptions

- **Baseline**: No eBPF instrumentation (native CUDA performance)
- **Empty probe**: Empty eBPF probe (minimal eBPF infrastructure overhead)
- **Entry probe**: eBPF probe attached to CUDA kernel entry point
- **Exit probe**: eBPF probe attached to CUDA kernel exit point
- **Entry+Exit**: eBPF probes attached to both entry and exit points
- **GPU Ringbuf**: GPU-side ring buffer for event logging
- **Global timer**: Global GPU timer measurements
- **Per-GPU-thread array**: Per-thread array lookups from GPU
- **Memtrace**: Memory access tracing
- **CPU Array/Hash map**: CPU-side map operations (update/lookup/delete)

