# Write and Run eBPF on GPU with bpftime

bpftime provides GPU support through its CUDA/ROCm attachment implementation, enabling eBPF programs to execute **within GPU kernels** on NVIDIA and AMD GPUs. This brings eBPF's programmability and observability to GPU computing workloads, enabling real-time profiling and debugging of GPU applications without source code modification.

> `Experimental`

## Architecture

### CUDA Attachment Pipeline

The GPU support is built on the `nv_attach_impl` system (`attach/nv_attach_impl/`), which implements an instrumentation pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application Process                         │
│                                                                   │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   CUDA App   │────▶│  bpftime     │────▶│  GPU Kernel  │    │
│  │              │     │  Runtime     │     │  with eBPF   │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                              │                      │            │
│                              ▼                      ▼            │
│                       ┌──────────────┐      ┌──────────────┐   │
│                       │ Shared Memory│      │  GPU Memory  │   │
│                       │  (Host-GPU)  │      │   (IPC)      │   │
│                       └──────────────┘      └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Examples

Complete working examples with full source code, build instructions, and READMEs are available on GitHub:

- **[cuda-counter](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/cuda-counter)**: Basic probe/retprobe with timing measurements
- **[cuda-counter-gpu-array](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/cuda-counter-gpu-array)**: Per-thread counters using GPU array maps
- **[cuda-counter-gpu-ringbuf](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/cuda-counter-gpu-ringbuf)**: Event streaming with ringbuf maps
- **[rocm-counter](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/rocm-counter)**: AMD GPU instrumentation (experimental)

Each example includes CUDA/ROCm application source, eBPF probe programs, Makefile, and detailed usage instructions.

### Key Components

1. **CUDA Runtime Hooking**: Intercepts CUDA API calls using Frida-based dynamic instrumentation
2. **PTX Modification**: Converts eBPF bytecode to PTX (Parallel Thread Execution) assembly and injects it into GPU kernels
3. **Helper Trampoline**: Provides GPU-accessible helper functions for map operations, timing, and context access
4. **Host-GPU Communication**: Enables synchronous calls from GPU to host via pinned shared memory

### Attachment Types

bpftime supports three attachment types for GPU kernels (defined in `attach/nv_attach_impl/nv_attach_impl.hpp:33-34`):

- **`ATTACH_CUDA_PROBE` (8)**: Executes eBPF code at kernel entry
- **`ATTACH_CUDA_RETPROBE` (9)**: Executes eBPF code at kernel exit
- **Memory capture probe (`__memcapture`)**: Special probe type for capturing memory access patterns

All types support specifying target kernel functions by name (e.g., `_Z9vectorAddPKfS0_Pf` for mangled C++ names).

## GPU-Specific BPF Maps

bpftime includes specialized map types optimized for GPU operations:

### `BPF_MAP_TYPE_NV_GPU_ARRAY_MAP` (1502)

GPU-resident array maps with **per-thread storage** for high-performance data collection.

**Key Features:**
- Data stored directly in GPU memory (CUDA IPC shared memory)
- Each thread gets isolated storage (`max_entries × max_thread_count × value_size`)
- Zero-copy access from GPU, DMA transfers to host
- Supports `bpf_map_lookup_elem()` and `bpf_map_update_elem()` in GPU code

**Implementation:** `runtime/src/bpf_map/gpu/nv_gpu_array_map.cpp:14-81`

### `BPF_MAP_TYPE_NV_GPU_RINGBUF_MAP` (1527)

GPU ring buffer maps for efficient **per-thread event streaming** to host.

**Key Features:**
- Lock-free per-thread ring buffers in GPU memory
- Variable-size event records with metadata
- Asynchronous data collection with low overhead
- Compatible with `bpf_perf_event_output()` helper

**Implementation:** `runtime/src/bpf_map/gpu/nv_gpu_ringbuf_map.cpp`

## GPU Helper Functions

bpftime provides GPU-specific eBPF helpers accessible from CUDA kernels (`attach/nv_attach_impl/trampoline/default_trampoline.cu:331-390`):

### Core GPU Helpers

| Helper ID | Function Signature | Description |
|-----------|-------------------|-------------|
| **501** | `ebpf_puts(const char *str)` | Print string from GPU kernel to host console |
| **502** | `bpf_get_globaltimer(void)` | Read GPU global timer (nanosecond precision) |
| **503** | `bpf_get_block_idx(u64 *x, u64 *y, u64 *z)` | Get CUDA block indices (blockIdx) |
| **504** | `bpf_get_block_dim(u64 *x, u64 *y, u64 *z)` | Get CUDA block dimensions (blockDim) |
| **505** | `bpf_get_thread_idx(u64 *x, u64 *y, u64 *z)` | Get CUDA thread indices (threadIdx) |
| **506** | `bpf_gpu_membar(void)` | Execute GPU memory barrier (`membar.sys`) |

### Standard BPF Helpers (GPU-Compatible)

The following standard eBPF helpers work on GPU with special optimizations:

- **`bpf_map_lookup_elem()`** (1): Fast path for GPU array maps, fallback to host for others
- **`bpf_map_update_elem()`** (2): Fast path for GPU array maps, fallback to host for others
- **`bpf_map_delete_elem()`** (3): Host call via shared memory
- **`bpf_trace_printk()`** (6): Formatted output to host console
- **`bpf_get_current_pid_tgid()`** (14): Returns host process PID/TID
- **`bpf_perf_event_output()`** (25): Optimized for GPU ringbuf maps

### Host-GPU Communication Protocol

For helpers requiring host interaction, bpftime uses a shared memory protocol with spinlocks and warp-level serialization for correctness. The protocol involves:

1. GPU thread acquires spinlock
2. Writes request parameters to shared memory
3. Sets flag and waits for host response
4. Host processes request and signals completion
5. GPU reads response and releases lock

## Building with GPU Support

### Prerequisites

- **NVIDIA CUDA Toolkit** (12.x recommended) or **AMD ROCm**
- **CMake** 3.15+
- **LLVM** 15+ (for PTX generation)
- **Frida-gum** for runtime hooking

### Build Configuration

```bash
# For NVIDIA CUDA
cmake -Bbuild \
  -DBPFTIME_ENABLE_CUDA_ATTACH=1 \
  -DBPFTIME_CUDA_ROOT=/usr/local/cuda-12.6 \
  -DCMAKE_BUILD_TYPE=Release

# For AMD ROCm (experimental)
cmake -Bbuild \
  -DBPFTIME_ENABLE_ROCM_ATTACH=1 \
  -DROCM_PATH=/opt/rocm

make -j$(nproc)
```

## References

1. [bpftime OSDI '25 Paper](https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng)
2. [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
3. [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
4. [eBPF Documentation](https://ebpf.io/)
5. [eGPU: Extending eBPF Programmability and Observability to GPUs](https://dl.acm.org/doi/10.1145/3723851.3726984)

Citation:

```
@inproceedings{yang2025egpu,
  title={eGPU: Extending eBPF Programmability and Observability to GPUs},
  author={Yang, Yiwei and Yu, Tong and Zheng, Yusheng and Quinn, Andrew},
  booktitle={Proceedings of the 4th Workshop on Heterogeneous Composable and Disaggregated Systems},
  pages={73--79},
  year={2025}
}
```

For questions or feedback, please open an issue on [GitHub](https://github.com/eunomia-bpf/bpftime) or [contact us](mailto:team@eunomia.dev).
