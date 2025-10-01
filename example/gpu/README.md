# Write and Run eBPF on GPU with bpftime

bpftime provides GPU support through its CUDA/ROCm attachment implementation, enabling eBPF programs to execute **within GPU kernels** on NVIDIA and AMD GPUs. This brings eBPF's programmability, observability, and customization capabilities to GPU computing workloads, enabling real-time profiling, debugging, and runtime extension of GPU applications without source code modification.

> `Experimental`

## Why eBPF on GPU?

GPUs have become the dominant accelerators for machine learning, scientific computing, and high-performance computing workloads, but their SIMT (Single Instruction, Multiple Thread) execution model introduces significant observability and extensibility challenges. Modern GPUs organize thousands of threads into warps (typically 32 threads) that execute in lockstep on streaming multiprocessors (SMs), with kernels launched asynchronously from the host. These threads navigate complex multi-level memory hierarchies—from fast but limited per-thread registers, to shared memory/LDS within thread blocks, through L1/L2 caches, to slower but abundant device memory—while contending with limited preemption capabilities that make kernel execution difficult to interrupt, inspect, or extend. This architectural complexity creates rich performance characteristics including warp divergence, memory coalescing patterns, bank conflicts, and occupancy variations that directly impact throughput, yet these behaviors remain largely opaque to traditional observability tools. Understanding and optimizing issues like kernel stalls, memory bottlenecks, inefficient synchronization, or suboptimal SM utilization requires fine-grained visibility into the execution flow, memory access patterns, and inter-warp coordination happening deep inside the GPU—along with the ability to dynamically inject custom logic—capabilities that existing tooling struggles to provide in a flexible, programmable manner.

Existing GPU tracing and profiling tools fall into two categories, each with significant limitations. First, many tracing tools operate exclusively at the CPU-GPU boundary by intercepting CUDA/ROCm userspace library calls (e.g., via LD_PRELOAD hooks on libcuda.so) or instrumenting kernel drivers at the system call layer. While this approach captures host-side events like kernel launches, memory transfers, and API timing, it fundamentally treats the GPU as a black box—providing no visibility into what happens during kernel execution, no correlation with specific warp behaviors or memory stalls, and no ability to adaptively modify behavior based on runtime conditions inside the device. Second, GPU vendor-specific profilers (NVIDIA's CUPTI, Nsight Compute, Intel's GTPin, AMD's ROCProfiler, research tools like NVBit or Neutrino) do provide device-side instrumentation and can collect hardware performance counters, warp-level traces, or instruction-level metrics. However, these tools operate in isolated ecosystems disconnected from Linux kernel observability and extension stacks: they cannot correlate GPU events with CPU-side eBPF probes (kprobes, uprobes, tracepoints), require separate data collection pipelines and analysis workflows, often impose substantial overhead (10-100x slowdowns for fine-grained instrumentation), and lack the dynamic programmability and control from the control plane that makes eBPF powerful for customizing what data to collect and how to process it in production environments, without recompilation or service interruption.

**bpftime's approach** bridges this gap by extending eBPF's programmability and customization model directly into GPU execution contexts, enabling eBPF programs to run natively inside GPU kernels alongside application workloads. The system defines a comprehensive set of GPU-side attach points that mirror the flexibility of CPU-side kprobes/uprobes: developers can instrument CUDA/ROCm device function entry and exit points (analogous to function probes), thread block lifecycle events (block begin/end), synchronization primitives (barriers, atomics), memory operations (loads, stores, transfers), and stream/event operations. eBPF programs written in restricted C are compiled through LLVM into device-native bytecode—PTX (Parallel Thread Execution) assembly for NVIDIA GPUs or SPIR-V for AMD/Intel—and dynamically injected into target kernels at runtime through binary instrumentation, without requiring source code modification or recompilation. The runtime provides a full eBPF execution environment on the GPU including: (1) a safety verifier adapted from PREVAIL to ensure bounded execution and memory safety in the SIMT context, (2) a rich set of GPU-aware helper functions for accessing thread/block/grid context, timing, synchronization, and formatted output, (3) specialized BPF map types that live in GPU memory for high-throughput per-thread data collection (GPU array maps) and event streaming (GPU ringbuf maps), and (4) a host-GPU communication protocol using shared memory and spinlocks for safely calling host-side helpers when needed. This architecture enables not only collecting fine-grained telemetry (per-warp timing, memory access patterns, control flow divergence) at nanosecond granularity, but also adaptively modifying kernel behavior based on runtime conditions, building custom extensions and optimizations, and unifying GPU observability with existing CPU-side eBPF programs into a single analysis pipeline—all while maintaining production-ready overhead characteristics. This enables:

- **3-10x faster performance** than tools like NVBit for instrumentation
- **Vendor-neutral design** that works across NVIDIA and AMD GPUs
- **Unified observability and control** with Linux kernel eBPF programs (kprobes, uprobes)
- **Fine-grained profiling and runtime customization** at the warp or instruction level
- **Adaptive GPU kernel memory optimization** and programmable scheduling across SMs
- **Dynamic extensions** for GPU workloads without recompilation
- **Accelerated eBPF applications** by leveraging GPU compute power

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
- **[kernelretsnoop](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/kernelretsnoop)**: Captures per-thread exit timestamps to detect thread divergence, memory access patterns, and warp scheduling issues
- **[threadhist](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/threadhist)**: Per-thread execution histogram using GPU array maps to detect workload imbalance
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
