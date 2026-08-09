
# CUDA Benchmark Results

**Device:** NVIDIA GeForce RTX 5090  
**Timestamp:** 2025-10-17T00:08:27.685106  

## Workload Configuration

| Workload | Elements | Iterations | Threads | Blocks |
|----------|----------|------------|---------|--------|
| large | benchmark/gpu/workload/vec_add | 100000 | 1000 | 512 |
| medium | benchmark/gpu/workload/vec_add | 10000 | 10000 | 256 |
| minimal | benchmark/gpu/workload/vec_add | 32 | 3 | 32 |
| small | benchmark/gpu/workload/vec_add | 1000 | 10000 | 256 |
| tiny | benchmark/gpu/workload/vec_add | 32 | 10000 | 32 |

## Benchmark Results

| Test Name | Workload | Avg Time (μs) | vs Baseline | Overhead |
|-----------|----------|---------------|-------------|----------|
| Baseline (minimal) | minimal | 5.94 | - | - |
| Baseline (tiny) | tiny | 5.21 | - | - |
| Baseline (small) | small | 5.27 | - | - |
| Baseline (medium) | medium | 5.32 | - | - |
| NVBit (minimal) | minimal | 10.16 | 5.94 | 1.71x (+71.0%) |
| NVBit (tiny) | tiny | 8.30 | 5.21 | 1.59x (+59.3%) |
| NVBit (small) | small | 8.25 | 5.27 | 1.57x (+56.5%) |
| NVBit (medium) | medium | 8.32 | 5.32 | 1.56x (+56.4%) |
| NVBit Tutorial - instr_count (minimal) | minimal | 414.82 | 5.94 | 69.84x (+6883.5%) |
| NVBit Tutorial - instr_count (tiny) | tiny | 69.58 | 5.21 | 13.36x (+1235.5%) |
| NVBit Tutorial - mem_trace (minimal) | minimal | 40.94 | 5.94 | 6.89x (+589.2%) |
| NVBit Tutorial - mem_trace (tiny) | tiny | 38.25 | 5.21 | 7.34x (+634.2%) |
| NVBit Tutorial - opcode_hist (minimal) | minimal | 323.54 | 5.94 | 54.47x (+5346.8%) |
| NVBit Tutorial - opcode_hist (tiny) | tiny | 71.99 | 5.21 | 13.82x (+1281.8%) |
| NVBit Tutorial - instr_count_bb (minimal) | minimal | 488.71 | 5.94 | 82.27x (+8127.4%) |
| NVBit Tutorial - instr_count_bb (tiny) | tiny | 13.20 | 5.21 | 2.53x (+153.4%) |
| NVBit Tutorial - instr_count_bb (small) | small | 63.49 | 5.27 | 12.05x (+1104.7%) |
| NVBit Tutorial - mem_printf2 (minimal) | minimal | 32.07 | 5.94 | 5.40x (+439.9%) |
| NVBit Tutorial - mem_printf2 (tiny) | tiny | 28.12 | 5.21 | 5.40x (+439.7%) |
| NVBit Tutorial - record_reg_vals (minimal) | minimal | FAILED | - | - |
| NVBit Tutorial - mov_replace (minimal) | minimal | 8.46 | 5.94 | 1.42x (+42.4%) |
| NVBit Tutorial - mov_replace (tiny) | tiny | 6.13 | 5.21 | 1.18x (+17.7%) |
| NVBit Tutorial - mov_replace (small) | small | 6.10 | 5.27 | 1.16x (+15.7%) |
| NVBit Tutorial - mov_replace (medium) | medium | 6.16 | 5.32 | 1.16x (+15.8%) |

