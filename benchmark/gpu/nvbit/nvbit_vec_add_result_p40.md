
# CUDA Benchmark Results

**Device:** Tesla P40  
**Timestamp:** 2025-10-18T01:22:19.702082  

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
| Baseline (minimal) | minimal | 15.48 | - | - |
| Baseline (tiny) | tiny | 20.65 | - | - |
| Baseline (small) | small | 14.15 | - | - |
| Baseline (medium) | medium | 14.35 | - | - |
| NVBit (minimal) | minimal | 40.99 | 15.48 | 2.65x (+164.8%) |
| NVBit (tiny) | tiny | 29.20 | 20.65 | 1.41x (+41.4%) |
| NVBit (small) | small | 24.90 | 14.15 | 1.76x (+76.0%) |
| NVBit (medium) | medium | 21.28 | 14.35 | 1.48x (+48.3%) |
| NVBit Tutorial - instr_count (minimal) | minimal | FAILED | - | - |
| NVBit Tutorial - instr_count (tiny) | tiny | FAILED | - | - |
| NVBit Tutorial - mem_trace (minimal) | minimal | FAILED | - | - |
| NVBit Tutorial - mem_trace (tiny) | tiny | FAILED | - | - |
| NVBit Tutorial - opcode_hist (minimal) | minimal | FAILED | - | - |
| NVBit Tutorial - opcode_hist (tiny) | tiny | FAILED | - | - |
| NVBit Tutorial - instr_count_bb (minimal) | minimal | 561.00 | 15.48 | 36.24x (+3524.0%) |
| NVBit Tutorial - instr_count_bb (tiny) | tiny | 30.36 | 20.65 | 1.47x (+47.0%) |
| NVBit Tutorial - instr_count_bb (small) | small | 67.28 | 14.15 | 4.75x (+375.5%) |
| NVBit Tutorial - mem_printf2 (minimal) | minimal | FAILED | - | - |
| NVBit Tutorial - mem_printf2 (tiny) | tiny | FAILED | - | - |
| NVBit Tutorial - record_reg_vals (minimal) | minimal | FAILED | - | - |
| NVBit Tutorial - mov_replace (minimal) | minimal | 32.28 | 15.48 | 2.09x (+108.5%) |
| NVBit Tutorial - mov_replace (tiny) | tiny | 25.75 | 20.65 | 1.25x (+24.7%) |
| NVBit Tutorial - mov_replace (small) | small | 22.95 | 14.15 | 1.62x (+62.2%) |
| NVBit Tutorial - mov_replace (medium) | medium | 19.04 | 14.35 | 1.33x (+32.7%) |

