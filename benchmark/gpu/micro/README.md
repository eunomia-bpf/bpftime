# micro benchmark

```
================================================================================
CUDA Benchmark Results
================================================================================

Device: NVIDIA GeForce RTX 5090
Timestamp: 2025-10-13T22:34:46.863972

Test Name                                Avg Time (μs)   Overhead       
----------------------------------------------------------------------
Baseline (no probe)                      5.49            -              
Baseline - Empty probe - 32 elem, 1 block × 32 threads 6.15            1.12x (+12.0%) 
Baseline - Empty probe - 1K elem, 4 blocks × 256 threads 6.25            1.14x (+13.8%) 
Baseline - Empty probe - 10K elem, 40 blocks × 256 threads 6.37            1.16x (+16.0%) 
Baseline - Empty probe - 100K elem, 196 blocks × 512 threads 6.65            1.21x (+21.1%) 
Baseline - Empty probe - 1M elem, 1954 blocks × 512 threads 10.80           1.97x (+96.7%) 
Entry probe - 32 elem, 1 block × 32 threads 6.11            1.11x (+11.3%) 
Entry probe - 1K elem, 4 blocks × 256 threads 6.33            1.15x (+15.3%) 
Entry probe - 10K elem, 40 blocks × 256 threads 6.34            1.15x (+15.5%) 
Entry probe - 1M elem, 1954 blocks × 512 threads 10.66           1.94x (+94.2%) 
Exit probe - 32 elem, 1 block × 32 threads 6.15            1.12x (+12.0%) 
Exit probe - 1K elem, 4 blocks × 256 threads 6.22            1.13x (+13.3%) 
Exit probe - 100K elem, 196 blocks × 512 threads 6.61            1.20x (+20.4%) 
Entry+Exit - 32 elem, 1 block × 32 threads 6.17            1.12x (+12.4%) 
Entry+Exit - 1K elem, 64 threads × 16 blocks 6.32            1.15x (+15.1%) 
Entry+Exit - 1K elem, 256 threads × 4 blocks 6.26            1.14x (+14.0%) 
Entry+Exit - 10K elem, 40 blocks × 256 threads 6.30            1.15x (+14.8%) 
Entry+Exit - 100K elem, 196 blocks × 512 threads 6.62            1.21x (+20.6%) 
Entry+Exit - 1M elem, 1954 blocks × 512 threads 10.80           1.97x (+96.7%) 
GPU Ringbuf - 32 elem, 1 block × 32 threads 10.50           1.91x (+91.3%) 
GPU Ringbuf - 1K elem, 16 blocks × 64 threads 16.35           2.98x (+197.8%)
GPU Ringbuf - 1K elem, 4 blocks × 256 threads 16.27           2.96x (+196.4%)
GPU Ringbuf - 10K elem, 40 blocks × 256 threads 35.62           6.49x (+548.8%)
GPU Ringbuf - 100K elem, 196 blocks × 512 threads 1.13            0.21x (+-79.4%)
GPU Ringbuf - 1M elem, 1954 blocks × 512 threads 1.12            0.20x (+-79.6%)
Global timer - 32 elem, 1 block × 32 threads 6.91            1.26x (+25.9%) 
Global timer - 1K elem, 4 blocks × 256 threads 7.11            1.30x (+29.5%) 
Global timer - 10K elem, 40 blocks × 256 threads 7.09            1.29x (+29.1%) 
Global timer - 1M elem, 1954 blocks × 512 threads 31.46           5.73x (+473.0%)
Per-GPU-thread array - 32 elem, 1 block × 32 threads 9.45            1.72x (+72.1%) 
Per-GPU-thread array - 1K elem, 16 blocks × 64 threads 9.74            1.77x (+77.4%) 
Per-GPU-thread array - 1K elem, 4 blocks × 256 threads 9.73            1.77x (+77.2%) 
Per-GPU-thread array - 10K elem, 40 blocks × 256 threads 10.14           1.85x (+84.7%) 
Per-GPU-thread array - 100K elem, 196 blocks × 512 threads 18.09           3.30x (+229.5%)
Per-GPU-thread array - 1M elem, 1954 blocks × 512 threads 1.07            0.19x (+-80.5%)
CPU Array map update - minimal, 1 block × 32 threads 33450.91        6093.06x (+609206.2%)
CPU Array map lookup - minimal, 1 block × 32 threads 33490.42        6100.26x (+609925.9%)
CPU Hash map update - minimal, 1 block × 32 threads 33463.71        6095.39x (+609439.3%)
CPU Hash map lookup - minimal, 1 block × 32 threads 34603.08        6302.93x (+630192.9%)
CPU Hash map delete - minimal, 1 block × 32 threads 33483.61        6099.02x (+609801.8%)

================================================================================
```