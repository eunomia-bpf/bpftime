# GPU Example Verifier Compatibility Analysis

This report reviews every `example/gpu/*/*.bpf.c` file against the current SIMT verifier rules:

- helper `505` (`bpf_get_thread_idx`) is lane-varying
- helper `511` (`bpf_get_lane_id`) is lane-varying
- helper `506` (`bpf_gpu_membar`) is prohibited
- conditional jumps should reject lane-varying branch conditions

## Key Finding

The current uniformity pass only marks helper return register `R0` as varying. It does not propagate lane-varying values written through helper pointer arguments onto the stack, and later `LDX` loads are treated as `UNIFORM`. In other words, the current risk in these examples is false negatives, not false positives.

| Example | Uses thread_idx | Uses lane_id | Varying branches | Expected verifier result | Notes |
| --- | --- | --- | --- | --- | --- |
| `atomizer/atomizer.bpf.c` | Yes | No | No | Pass | `thread_idx` is declared but unused. Control flow depends on `block_idx`, map lookup null checks, and partition bounds derived from block/grid coordinates. |
| `cuda-counter/cuda_probe.bpf.c` | No | No | No | Pass | Uses `block_idx` only. No lane-varying helper is used in control flow. |
| `cudagraph/cuda_probe.bpf.c` | Yes | No | Yes | Pass today | Leader-thread filter branches on `thread_idx == 0`. Source-level SIMT policy would reject it, but current stack-insensitive uniformity tracking likely misses it. |
| `cutlass/counter/cutlass_launch_counter.bpf.c` | No | No | No | Pass | Fixed-key counter update only. |
| `directly_run_on_gpu/directly_run.bpf.c` | Yes | No | Yes | Pass today | `cuda__vec_add` and `cuda__gemm` branch on indices derived from `thread_idx`. These are source-level varying branches that the current analysis likely accepts. |
| `faiss-test/threadhist.bpf.c` | No | No | No | Pass | Counter increment only. |
| `gpu_shared_map/gpu_shared_map.bpf.c` | Yes | No | Yes | Pass today | Both sections branch on `idx == 0`, where `idx` includes `thread_idx`. Likely false-negative acceptance today. |
| `host_map_test/host_map_test.bpf.c` | Yes | No | No | Pass | Uses `thread_idx` to build map keys and stored values, but not branch predicates. Current checks do not validate key pointee uniformity. |
| `kernel_trace/kernel_trace.bpf.c` | Yes | No | Yes | Pass today | Single-thread sampling filter depends on `thread_idx`. Divergent in source, but likely accepted by the current checker. |
| `kernelretsnoop/kernelretsnoop.bpf.c` | Yes | No | No | Pass | `thread_idx` is recorded into emitted data only. |
| `launchlate-kernel-gpu-shared-map/launchlate.bpf.c` | No | No | No | Pass | Mixed CPU `uprobe` and GPU `kprobe` file. GPU control flow depends on timestamps and map lookups, not lane-varying helpers. |
| `launchlate/launchlate.bpf.c` | No | No | No | Pass | Same as the kernel-shared-map variant, without `thread_idx`, `lane_id`, or helper `506`. |
| `llama-cpp-test/threadhist.bpf.c` | No | No | No | Pass | Counter increment only. |
| `mem_trace/mem_trace.bpf.c` | No | No | No | Pass | No lane-varying helper use. |
| `pytorch-test/threadhist.bpf.c` | No | No | No | Pass | Counter increment only. |
| `rocm-counter/rocm_probe.bpf.c` | No | No | No | Pass | Uses `block_idx` only; no varying helper in predicates. |
| `threadhist-gpu-kernel-shared-map/threadhist.bpf.c` | No | No | No | Pass | Contains one GPU `kretprobe` and one CPU `uretprobe`; neither uses `thread_idx`, `lane_id`, or `membar`. |
| `threadhist/threadhist.bpf.c` | No | No | No | Pass | Counter increment only. |
| `threadscheduling/threadscheduling.bpf.c` | Yes | Yes | No | Pass | Stores `thread_idx`, `sm_id`, `warp_id`, and `lane_id` into maps, but does not branch on them. |

## Summary

- Files using helper `505`: `atomizer`, `cudagraph`, `directly_run_on_gpu`, `gpu_shared_map`, `host_map_test`, `kernel_trace`, `kernelretsnoop`, `threadscheduling`
- Files using helper `511`: `threadscheduling` only
- Files using helper `506`: none
- Files with source-level lane-varying branches: `cudagraph`, `directly_run_on_gpu`, `gpu_shared_map`, `kernel_trace`
- Files likely accepted by the current implementation despite those source-level varying branches: the same four files above, due to missing propagation from helper out-parameters to later stack loads
