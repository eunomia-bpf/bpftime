# GPU Verifier Evaluation Summary

This file consolidates the regenerated Release-build evaluation artifacts in Markdown tables suitable for downstream LaTeX conversion.

## Key Findings

- RQ1 correctness: 10/10 rows matched the expected verdict.
- RQ2 example sweep: 13 pass, 5 true positive, and 0 false positive results across 18 `example/gpu/*/*.bpf.c` files.
- RQ3 performance: median total verifier time rises from 18.107 μs at size 16 to 1284.763 μs at size 2048; median PREVAIL time stays at 0.000 μs or below because `skip_prevail=true` is the default path.
- RQ6 comparison: the SIMT-aware verifier catches GPU-specific divergence, prohibited-helper, varying-atomic, varying-key, and helper-budget patterns that `no verification` misses, while standard PREVAIL design coverage remains focused on classic eBPF safety classes.

## Numbers To Cite

| Metric | Value |
| --- | --- |
| Correctness rows matching expectation | 10 / 10 |
| Example corpus size | 18 |
| Example corpus true positives | 5 |
| Example corpus false positives | 0 |
| Perf median total @ 16 instructions | 18.107 μs |
| Perf median total @ 2048 instructions | 1284.763 μs |
| Perf median SIMT pass @ 2048 instructions | 1283.665 μs |
| Largest median PREVAIL time in perf JSON | 0.000 μs |

## RQ1 - Correctness Table

| Program | LOC | Safety Property | Expected | Result | Time (μs) |
| --- | --- | --- | --- | --- | --- |
| varying_branch | 31 | Varying branch condition | REJECT | REJECT | 49 |
| prohibited_helper | 27 | Prohibited helper (membar) | REJECT | REJECT | 94 |
| varying_atomic | 30 | Varying atomic address | REJECT | REJECT | 45 |
| varying_map_key | 24 | Varying map key | REJECT | REJECT | 71 |
| resource_exceeded | 92 | Helper/resource budget exceeded | REJECT | REJECT | 313 |
| safe_counter | 31 | None (safe) | PASS | PASS | 36 |
| safe_block_idx_branch | 26 | None (safe block-uniform branch) | PASS | PASS | 52 |
| cuda-counter (entry) | 74 | None (real example) | PASS | PASS | 73 |
| cuda-counter (return) | 74 | None (real example) | PASS | PASS | 93 |
| directly_run_on_gpu (entry) | 128 | None (real example) | PASS | PASS | 159 |

## RQ2 - False Positive Analysis

| Example | Sections | Result | Errors | Classification |
| --- | --- | --- | --- | --- |
| atomizer/atomizer.bpf.c | kprobe/_Z9vectorAddPKfS0_Pf | PASS |  | pass |
| cuda-counter/cuda_probe.bpf.c | kprobe/_Z9vectorAddPKfS0_Pf, kretprobe/_Z9vectorAddPKfS0_Pf | PASS |  | pass |
| cudagraph/cuda_probe.bpf.c | kprobe/_Z9vectorAddPKfS0_Pf, kretprobe/_Z9vectorAddPKfS0_Pf | REJECT | kprobe/_Z9vectorAddPKfS0_Pf: Warp-Uniform Branch Conditions at instruction 25: branch predicate is lane-varying // kretprobe/_Z9vectorAddPKfS0_Pf: Warp-Uniform Branch Conditions at instruction 25: branch predicate is lane-varying | true positive |
| directly_run_on_gpu/directly_run.bpf.c | kprobe/__directly_run | PASS |  | pass |
| faiss-test/threadhist.bpf.c | kretprobe/_ZN5faiss3gpu14l2NormRowMajorIf6float4Li8ELb1EEEvNS0_6TensorIT0_Li2ELb1ElNS0_6traits16DefaultPtrTraitsEEENS3_IfLi1ELb1ElS6_EE | PASS |  | pass |
| gpu_shared_map/gpu_shared_map.bpf.c | kretprobe/_Z9vectorAddPKfS0_Pf, kprobe/_Z9vectorAddPKfS0_Pf | REJECT | kretprobe/_Z9vectorAddPKfS0_Pf: Warp-Uniform Branch Conditions at instruction 46: branch predicate is lane-varying // kprobe/_Z9vectorAddPKfS0_Pf: Warp-Uniform Branch Conditions at instruction 32: branch predicate is lane-varying | true positive |
| host_map_test/host_map_test.bpf.c | kprobe/_Z9vectorAddPKfS0_Pf, kretprobe/_Z9vectorAddPKfS0_Pf | REJECT | kprobe/_Z9vectorAddPKfS0_Pf: Map Update Key Uniformity at instruction 40: map key bytes are lane-varying | true positive |
| kernel_trace/kernel_trace.bpf.c | kprobe/_Z9vectorAddPKfS0_Pfi | REJECT | kprobe/_Z9vectorAddPKfS0_Pfi: Warp-Uniform Branch Conditions at instruction 33: branch predicate is lane-varying | true positive |
| kernelretsnoop/kernelretsnoop.bpf.c | kretprobe/_Z9vectorAddPKfS0_Pf | PASS |  | pass |
| launchlate/launchlate.bpf.c | uprobe, kprobe/_Z9vectorAddPKfS0_Pf | PASS |  | pass |
| launchlate-kernel-gpu-shared-map/launchlate.bpf.c | uprobe, kprobe/cuda__Z9vectorAddPKfS0_Pf | PASS |  | pass |
| llama-cpp-test/threadhist.bpf.c | kretprobe/_Z12rms_norm_f32ILi1024ELb1ELb0EEvPKfPfilllfS1_lll5uint3S3_S3_S3_S1_lllS3_S3_S3_S3_ | PASS |  | pass |
| mem_trace/mem_trace.bpf.c | kprobe/__memcapture | PASS |  | pass |
| pytorch-test/threadhist.bpf.c | kretprobe/_ZN2at6native20bitonicSortKVInPlaceILin2ELin1ELi16ELi16EilNS0_4LTOpIiLb1EEEjEEvNS_4cuda6detail10TensorInfoIT3_T6_EES8_S8_S8_NS6_IT4_S8_EES8_T5_ | PASS |  | pass |
| rocm-counter/rocm_probe.bpf.c | kprobe/_Z9vectorAddPKfS0_Pf, kretprobe/_Z9vectorAddPKfS0_Pf | PASS |  | pass |
| threadhist/threadhist.bpf.c | kretprobe/_Z9vectorAddPKfS0_Pf | PASS |  | pass |
| threadhist-gpu-kernel-shared-map/threadhist.bpf.c | kretprobe/cuda__Z9vectorAddPKfS0_Pf, uretprobe/./vec_add:_Z16cudaLaunchKernelIcE9cudaErrorPT_4dim3S3_PPvmP11CUstream_st | PASS |  | pass |
| threadscheduling/threadscheduling.bpf.c | kprobe/_Z9vectorAddPKfS0_Pf | REJECT | kprobe/_Z9vectorAddPKfS0_Pf: Map Update Key Uniformity at instruction 93: map key bytes are lane-varying | true positive |

## RQ3 - Performance Breakdown

| Size | Total (μs) | SIMT Pass (μs) |
| ---: | ---: | ---: |
| 16 | 18.107 | 17.172 |
| 32 | 28.989 | 28.090 |
| 64 | 49.397 | 48.546 |
| 128 | 89.059 | 87.999 |
| 256 | 174.248 | 173.165 |
| 512 | 345.887 | 344.748 |
| 1024 | 639.467 | 638.407 |
| 2048 | 1284.763 | 1283.665 |

## RQ6 - Comparison Table

| Pattern | Representative Input | No Verification | Standard PREVAIL (design) | SIMT-aware (measured) |
| --- | --- | --- | --- | --- |
| Varying branch condition | gpu_unsafe_programs/varying_branch.bpf.c | MISS | MISS | CATCH |
| Prohibited helper (membar) | gpu_unsafe_programs/prohibited_helper.bpf.c | MISS | MISS | CATCH |
| Varying atomic address | gpu_unsafe_programs/varying_atomic.bpf.c | MISS | MISS | CATCH |
| Varying map key | gpu_unsafe_programs/varying_map_key.bpf.c | MISS | MISS | CATCH |
| Helper-call budget exceeded | gpu_unsafe_programs/resource_exceeded.bpf.c | MISS | MISS | CATCH |
| Memory safety (null deref) | builtin:null_deref | MISS | CATCH | MISS |
| Division by zero | builtin:division_by_zero | MISS | CATCH | MISS |
| Unbounded loop (self-loop) | builtin:resource_exceeded | MISS | CATCH | MISS |

Notes:
- `No Verification` is the baseline with the verifier disabled, so unsafe programs are not intercepted.
- `Standard PREVAIL (design)` is design-based coverage, not a measurement on GPU object files.
- `Unbounded loop (self-loop)` uses builtin `resource_exceeded`; it is distinct from `gpu_unsafe_programs/resource_exceeded.bpf.c`, which models helper-call budget exhaustion.

Examples classified as `true positive`: `cudagraph/cuda_probe.bpf.c`, `gpu_shared_map/gpu_shared_map.bpf.c`, `host_map_test/host_map_test.bpf.c`, `kernel_trace/kernel_trace.bpf.c`, `threadscheduling/threadscheduling.bpf.c`.
