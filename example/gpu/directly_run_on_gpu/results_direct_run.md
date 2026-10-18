### eBPF Direct-run vs Native CUDA (latest)

**Device**: Tesla P40  
**Timestamp**: 2025-10-21

#### VecAdd (minimal preset)

| Test | Avg Time (μs) | Baseline | Overhead |
|---|---:|---:|---:|
| Baseline (minimal, CUDA kernel) | 16.78 | - | - |
| eBPF Direct Run VecAdd (minimal) | 27.58 | 16.78 | 1.64x (+64.4%) |

Note: Based on latest `benchmark/gpu/micro/examples_vec_add_result.md` (manual run path in README).

#### GEMM (minimal preset)

| Test | Avg Time (μs) | Baseline | Overhead |
|---|---:|---:|---:|
| Baseline (minimal, CUDA kernel) | 2255.38 | - | - |
| eBPF Direct Run GEMM (minimal) | 2336.83 | 2255.38 | 1.04x (+3.6%) |

Note: Data from `benchmark/gpu/micro/micro_gemm_result.md` minimal case (32×32×32, grid=2×2, block=16×16; direct-run script: `benchmark/gpu/micro/run_direct_gemm.sh`).


