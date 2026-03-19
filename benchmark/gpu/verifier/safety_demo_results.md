# RQ4: GPU Safety Demonstration

Date: 2026-03-18

## Environment

- GPU: NVIDIA GeForce RTX 5090
- Driver: 575.57.08
- CUDA toolkit: 12.9.86 at `/usr/local/cuda-12.9`
- Verifier build: `-DENABLE_EBPF_VERIFIER=ON`
- CUDA attach build: `-DBPFTIME_ENABLE_CUDA_ATTACH=ON`

## Build Status

The project does build with CUDA support on this machine, but the working configure line needed the CUDA root pinned explicitly:

```bash
cmake -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_EBPF_VERIFIER=ON \
  -DBPFTIME_ENABLE_CUDA_ATTACH=ON \
  -DBPFTIME_CUDA_ROOT=/usr/local/cuda-12.9
cmake --build build -j"$(nproc)"
make -C example/gpu/cuda-counter
```

Practical caveats encountered during bring-up:

- A stale build cache pointed to `/usr/local/cuda-12.6`; that version was missing the CUDA pieces needed here, including `nvPTXCompiler.h`.
- `third_party/ubpf/external/bpf_conformance` had to be initialized:

```bash
git -C third_party/ubpf submodule update --init external/bpf_conformance
```

- After the build was fixed, the standalone GPU verifier test binaries passed:

```text
./build/bpftime-verifier/bpftime_gpu_verifier_tests --reporter compact
All tests passed (25 assertions in 4 test cases)

./build/bpftime-verifier/bpftime_gpu_verifier_e2e_tests --reporter compact
All tests passed (30 assertions in 4 test cases)
```

## Status Of The Shipped CUDA Example

`example/gpu/cuda-counter` builds, and the normal `cuda_probe` plus `vec_add` path now reaches the GPU verifier and PTX pipeline. On this branch it is still not a clean success case:

- `cuda__probe` verification: 16.130 ms
- `cuda__retprobe` verification: 40.231 ms
- PTX compile: 64.640 ms
- PTX patch/load: 969.164 ms
- Final outcome: `CUDA_ERROR_ILLEGAL_ADDRESS`

So the packaged example is useful as a sanity check that the CUDA attach pipeline is alive, but it is not stable enough to serve as the safety-demo artifact.

## Safety Demo Setup

I used a minimal live loader plus two single-program CUDA eBPF objects:

- Safe object: `benchmark/gpu/verifier/.live_demo/live_safe_block_idx.bpf.o`
  - Branch depends on `blockIdx.x`, which is warp-uniform.
- Unsafe object: `benchmark/gpu/verifier/.live_demo/live_unsafe_varying_branch.bpf.o`
  - Branch depends on `threadIdx.x`, which is lane-varying.

Launch pattern:

```bash
# server side: load the BPF object and attach it
BPFTIME_VERIFIER_LEVEL=NO_VERIFY \
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
benchmark/gpu/verifier/.live_demo/live_object_loader \
benchmark/gpu/verifier/.live_demo/live_safe_block_idx.bpf.o 15

# client side: run the CUDA workload with the agent
BPFTIME_VERIFIER_LEVEL=STRICT \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
example/gpu/cuda-counter/vec_add
```

The server was intentionally pinned to `NO_VERIFY` so that the live demo exercised the GPU-side verifier in the client attach path rather than the generic userspace verifier.

## Results

| Case | Client verifier mode | Observed behavior |
| --- | --- | --- |
| Safe object | `STRICT` | Accepted and injected successfully. The client logged `GPU verifier elapsed for cuda__live_safe: 2.825 ms`, `PTX compile elapsed: 16.962 ms`, `PTX patch/load elapsed: 453.469 ms`, and then kept producing correct outputs such as `counter = 234 (expected 234)` and `C[1] = 3 (expected 3)`. |
| Unsafe object | `NO_VERIFY` | The client logged `Skipping GPU eBPF verification for cuda__live_unsa because BPFTIME_VERIFIER_LEVEL=NO_VERIFY`, then `PTX compile elapsed: 19.630 ms`, `PTX patch/load elapsed: 462.585 ms`, and the workload continued to run. |
| Unsafe object | `STRICT` | The verifier rejected the program at load time with `GPU eBPF verification failed for cuda__live_unsa: Warp-Uniform Branch Conditions at instruction 13: branch predicate is lane-varying`. Verification itself took 2.766 ms. |

Relevant logs from this run:

- Safe strict pass: `/tmp/bpftime_safe_strict_client_clean.log`
- Unsafe with verifier disabled: `/tmp/bpftime_noverify_unsafe_client_clean.log`
- Unsafe with strict verifier: `/tmp/bpftime_strict_unsafe_client_clean.log`

## What This Demonstrates

1. A safe CUDA eBPF program does load and run correctly with the GPU verifier enabled in `STRICT` mode.
2. An unsafe CUDA eBPF program with a lane-varying branch is allowed to load when verification is disabled.
3. The same unsafe program is rejected by the GPU verifier in `STRICT` mode for the intended SIMT-specific reason: non-warp-uniform control flow.

This is the core safety claim for RQ4: the GPU verifier catches a class of unsafe programs that would otherwise reach the live CUDA attach path.

## Current Runtime Bug

The strict rejection path is not fail-closed yet.

After rejecting the unsafe program, the runtime still continues far enough to hit:

```text
Failed to instantiate handler 19
what():  shared_mem_ptr is not initialized before loading PTX
```

The client exits with `134` (`Aborted`). That does **not** invalidate the safety result, because the unsafe program was already rejected before injection, but it is an important runtime bug:

- intended behavior: stop cleanly after strict verifier rejection
- current behavior: reject, then continue into later CUDA attach code and abort

For a paper/demo write-up, the right interpretation is:

- verifier behavior: correct
- runtime error handling after verifier rejection: currently incorrect
