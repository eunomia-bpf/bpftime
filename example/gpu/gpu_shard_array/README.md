# GPU Shard Array Map Example

This example demonstrates the behavior of `BPF_MAP_TYPE_GPU_ARRAY_MAP` (a normal, non-per-thread, single-copy shared array) in user space in conjunction with bpftime. This example relies on the agent side for updates, while the server only reads and does not write back to the HOST.

-   **BPF Program:** `gpu_shard_array.bpf.c` increments `key=0` at the `kretprobe/vectorAdd` hook.
-   **Host Program:** `gpu_shard_array.c` periodically reads and prints `counter[0]`.
-   **CUDA Program:** `vec_add.cu` runs periodically, triggering the BPF program to update the map.

## Prerequisites
-   clang/llvm, make, and gcc are installed.
-   A working CUDA environment (`nvcc`, `libcudart`) is available, and a GPU is accessible.
-   Repository submodules `third_party/libbpf` and `third_party/bpftool` are available (the example's Makefile will automatically build the bootstrap bpftool and a static libbpf).
-   Recommended: bpftime has been built at the top level (with CUDA attach enabled):
    -   `cmake -B build -S . -DBPFTIME_ENABLE_CUDA_ATTACH=ON -DBPFTIME_CUDA_ROOT=/usr/local/cuda`
    -   `cmake --build build -j`

## Build
```bash
cd example/gpu/gpu_shard_array
make
```
This generates:
-   `gpu_shard_array` (the host program)
-   `vec_add` (the CUDA test kernel)
-   `.output/` (intermediate products and the BPF skeleton)

## Running Steps (Agent/Server Mode)
1) Start the server (read-only process):
```bash
/bpftime/build/tools/cli/bpftime load /bpftime/example/gpu/gpu_shard_array/gpu_shard_array
```
2) Start the agent's target program (loads CUDA attach, triggers BPF):
```bash
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  BPFTIME_LOG_OUTPUT=console SPDLOG_LEVEL=debug example/gpu/gpu_shard_array/vec_add
```
3) Expected Output:
-   **Server:** Periodically prints `counter[0]=N`, which should monotonically increase (subject to concurrent overwrites, but the trend is upward).
-   **Agent:** You should see `CUDA Received call request id ...` and related logs (logs will be significantly reduced when not using the CPU handshake path).

## Consistency Explanation (Brief)
-   **Writes:** The device side performs a `memcpy` overwrite. After the write, a system-level memory fence is executed (implemented in the trampoline) to ensure visibility to the host.
-   **Reads:** Reads are lock-free. It is possible to read a slightly older value, but you will not read a partially written value.
-   For "stronger read consistency," locks or double-read version checks can be added on the reading side (this is not enabled in this example).

## Technical Points of Non-Per-Thread GPU Map (Current Implementation)
-   Goal and Form:
    -   **Non-per-thread**: The GPU and host share a single copy of the map (array); different threads no longer have independent copies.
    -   **UVA Zero-Copy**: Shared memory on the host side is registered by the agent, allowing both the GPU and the host to see the same data via UVA (Unified Virtual Addressing).
-   Update Semantics:
    -   Direct device-side writes (trampoline fast-path) use a `memcpy` overwrite, which is non-atomic; the last writer wins. A system fence after the write guarantees visibility.
    -   This approach no longer relies on a CPU handshake, reducing latency and overhead. For precise counting, you should aggregate at a higher level or shard keys to reduce conflicts.

## Example Write Method (Update)
-   **BPF Side** (`gpu_shard_array.bpf.c`):
    -   In `kretprobe/vectorAdd`, the program reads `counter[0]`, adds 1, and then uses `BPF_ANY` to overwrite the value (the `memcpy` is executed on the device).

## Troubleshooting
-   **No `nvcc`:** The script will prompt you to skip the CUDA build; the example will not be able to trigger the counter increment.
-   **Permissions/Dependencies:** Confirm that clang, make, and gcc are installed, and that libbpf/bpftool can be built from the `third_party/` directory.
-   **CUDA Driver:** You must be able to run a minimal CUDA kernel; otherwise, `vec_add` will fail.