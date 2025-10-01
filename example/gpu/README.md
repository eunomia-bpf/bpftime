# Write and Run eBPF on GPU with bpftime

bpftime provides GPU support through its CUDA/ROCm attachment implementation, enabling eBPF programs to execute **within GPU kernels** on NVIDIA and AMD GPUs. This brings eBPF's programmability, observability, and customization capabilities to GPU computing workloads, enabling real-time profiling, debugging, and runtime extension of GPU applications without source code modification.

> **Note:** GPU support is still experimental. For questions or suggestions, [open an issue](https://github.com/eunomia-bpf/bpftime/issues) or [contact us](mailto:team@eunomia.dev).

## The Problem: GPU Observability Challenges

GPUs have become the dominant accelerators for machine learning, scientific computing, and high-performance computing workloads, but their SIMT (Single Instruction, Multiple Thread) execution model introduces significant observability and extensibility challenges. Modern GPUs organize thousands of threads into warps (typically 32 threads) that execute in lockstep on streaming multiprocessors (SMs), with kernels launched asynchronously from the host. These threads navigate complex multi-level memory hierarchies—from fast but limited per-thread registers, to shared memory/LDS within thread blocks, through L1/L2 caches, to slower but abundant device memory—while contending with limited preemption capabilities that make kernel execution difficult to interrupt, inspect, or extend. This architectural complexity creates rich performance characteristics including warp divergence, memory coalescing patterns, bank conflicts, and occupancy variations that directly impact throughput, yet these behaviors remain largely opaque to traditional observability tools. Understanding and optimizing issues like kernel stalls, memory bottlenecks, inefficient synchronization, or suboptimal SM utilization requires fine-grained visibility into the execution flow, memory access patterns, and inter-warp coordination happening deep inside the GPU—along with the ability to dynamically inject custom logic—capabilities that existing tooling struggles to provide in a flexible, programmable manner.

Existing GPU tracing and profiling tools fall into two categories, each with significant limitations. First, many tracing tools operate exclusively at the CPU-GPU boundary by intercepting CUDA/ROCm userspace library calls (e.g., via LD_PRELOAD hooks on libcuda.so) or instrumenting kernel drivers at the system call layer. While this approach captures host-side events like kernel launches, memory transfers, and API timing, it fundamentally treats the GPU as a black box—providing no visibility into what happens during kernel execution, no correlation with specific warp behaviors or memory stalls, and no ability to adaptively modify behavior based on runtime conditions inside the device. Second, GPU vendor-specific profilers (NVIDIA's CUPTI, Nsight Compute, Intel's GTPin, AMD's ROCProfiler, research tools like NVBit or Neutrino) do provide device-side instrumentation and can collect hardware performance counters, warp-level traces, or instruction-level metrics. However, these tools operate in isolated ecosystems disconnected from Linux kernel observability and extension stacks: they cannot correlate GPU events with CPU-side eBPF probes (kprobes, uprobes, tracepoints), require separate data collection pipelines and analysis workflows, often impose substantial overhead (10-100x slowdowns for fine-grained instrumentation), and lack the dynamic programmability and control from the control plane that makes eBPF powerful for customizing what data to collect and how to process it in production environments, without recompilation or service interruption.

### Timeline Visibility Gap

#### Understanding the GPU Execution Model

When you run a CUDA application, a typical workflow begins with the host (CPU) allocating memory on the device (GPU), followed by data transfer from host memory to device memory, then GPU kernels (functions) are launched to process the data, after which results are transferred back from device to host, and finally device memory is freed.

Each operation in this process involves CUDA API calls, such as `cudaMalloc()` for memory allocation, `cudaMemcpy()` for data transfer, and `cudaLaunchKernel()` for kernel execution. These operations can be executed in two modes:

**Synchronous mode**: Each API call blocks the CPU thread until the GPU operation completes. For example, `cudaMemcpy()` will not return until the data transfer is finished, and calling `cudaDeviceSynchronize()` after a kernel launch forces the CPU to wait until kernel execution completes. This mode is simpler to understand and debug, but it prevents overlapping CPU work with GPU execution, potentially leaving hardware idle.

**Asynchronous mode**: API calls like `cudaMemcpyAsync()` and `cudaLaunchKernel()` return immediately after enqueuing work to a CUDA stream, allowing the CPU to continue executing while the GPU processes data in parallel. Synchronization points like `cudaStreamSynchronize()` or `cudaEventSynchronize()` are used only when the CPU needs to wait for GPU results. This mode enables maximum hardware utilization by overlapping computation with data transfer and allowing the CPU to prepare future work while the GPU is busy, but it makes the execution timeline more complex.

Both modes follow the same underlying workflow—allocate, transfer, compute, transfer back, free—but asynchronous execution introduces temporal decoupling between when operations are *enqueued* (CPU-side API call) and when they actually *execute* (GPU-side hardware activity). This decoupling is the root cause of the observability challenge we'll explore next.

#### The Visibility Problem

The discrepancy between CPU-visible and GPU-actual timelines illustrates the fundamental observability challenge with GPU workloads. Traditional profiling tools that operate at the CPU-GPU boundary can only observe host-side API calls and synchronization points, treating the GPU as an opaque black box. This creates a critical gap between what developers can measure and what actually happens during execution, leading to missed optimization opportunities and misdiagnosed performance issues.

#### Synchronous Kernel Execution

Consider the typical synchronous workflow: allocate device memory with `cudaMalloc()`, copy data to GPU with `cudaMemcpy()`, launch a kernel, synchronize with `cudaDeviceSynchronize()`, copy results back, and free memory. In synchronous mode, each operation blocks the CPU until completion:

```
CPU Timeline (what traditional tools see):
───────────────────────────────────────────────────────────────────────────
 cudaMalloc()      cudaMemcpy()       cudaLaunchKernel()  cudaDeviceSync()
      ↓                 ↓                     ↓                  ↓
──────●─────────────────●─────────────────────●──────────────────●─────────
      ↑                 ↑                     ↑                  ↑
   returns           returns               returns            returns
   ~1μs later        after 200μs           immediately        after kernel
                     (H→D done)            (enqueued)         completes

GPU Timeline (actual execution with hidden phases):
───────────────────────────────────────────────────────────────────────────
      ◄─Alloc─►◄────H→D DMA────►◄──Launch──►◄──Kernel Exec──►◄─Cleanup─►
      │ ~1μs  │     200μs        │   5μs    │     100μs       │  ~10μs  │
──────┴───────┴──────────────────┴──────────┴─────────────────┴─────────┴──
                                              ↑                           ↑
                                           SM busy                    SM idle
```

From the CPU perspective, traditional profiling tools only observe discrete API call events and when they return. For memory allocation (`cudaMalloc()`), the call returns almost instantly since it only reserves address space. For `cudaMemcpy()`, the call blocks until the host-to-device (H→D) DMA transfer completes—but tools only see the total blocking time, not the actual PCIe transfer characteristics. For kernel launch (`cudaLaunchKernel()`) followed by `cudaDeviceSynchronize()`, the tools see two events: the launch call that enqueues work, and the synchronization call that blocks the CPU thread until GPU completion. The time between these calls appears as a single opaque duration—the tools cannot distinguish between kernel launch overhead, actual computation time, and cleanup overhead.

However, the GPU timeline reveals a much more complex execution sequence. After the launch API call returns, there is a **kernel launch overhead** period (typically 5-50 microseconds) where the CUDA runtime prepares the kernel for execution: it validates parameters, allocates kernel argument memory, configures the grid/block dimensions, and schedules the kernel on the GPU's hardware queue. Only after this setup completes does the actual **kernel execution** phase begin, where streaming multiprocessors (SMs) execute the parallel workload. During this phase, warps of threads execute instructions, access memory hierarchies, and synchronize within thread blocks—but all of this internal behavior is completely invisible to CPU-side tools. Finally, there is a **kernel exit and cleanup** phase where SMs complete their work, write results back through the memory hierarchy, and signal completion to the runtime.

This visibility gap has critical implications for performance analysis. If a kernel appears slow from the CPU perspective, developers cannot determine whether the issue is launch overhead (indicating too many small kernel calls that should be batched), actual compute inefficiency (suggesting algorithmic improvements or memory access optimization), or SM underutilization (indicating poor grid/block configuration). Without GPU-side instrumentation, these distinct phases are conflated into a single measurement, making targeted optimization impossible.

#### Asynchronous Kernel Execution

The observability challenge becomes even more severe with asynchronous operations, which are essential for achieving high GPU utilization through overlapping computation and data transfer. Let's examine the same complete workflow (allocate → transfer H→D → compute → transfer D→H), but using asynchronous APIs:

```
CPU Timeline (what traditional tools see):
─────────────────────────────────────────────────────────────────────────────────
 cudaMalloc()  cudaMemcpyAsync() cudaLaunchKernel() cudaMemcpyAsync() cudaStreamSync()
      ↓               ↓                  ↓                 ↓                ↓
──────●───────────────●──────────────────●─────────────────●────────────────●────────
      ↑               ↑                  ↑                 ↑                ↑
   returns         returns            returns           returns         returns
   ~1μs later      immediately        immediately       immediately     after all
   (blocked)       (enqueued)         (enqueued)        (enqueued)      work done
                   CPU continues→     CPU continues→    CPU continues→  (blocked)

GPU Timeline (actual execution - sequential in stream):
─────────────────────────────────────────────────────────────────────────────────
      ◄─Alloc─►◄───H→D DMA────►◄Launch►◄──Kernel Exec──►◄───D→H DMA────►
      │ ~1μs  │     200μs       │ 5μs  │     100μs       │     150μs     │
──────┴───────┴─────────────────┴──────┴─────────────────┴───────────────┴─────────
                ↑                         ↑                               ↑
           CPU already                GPU compute                    CPU still
           moved on                   happening                      elsewhere
```

In this common asynchronous pattern, the application follows the same workflow as synchronous mode but uses async APIs to avoid blocking. First, `cudaMalloc()` still blocks briefly (~1μs) to allocate device memory. Then `cudaMemcpyAsync()` enqueues the host-to-device (H→D) data transfer and returns immediately. Next, `cudaLaunchKernel()` enqueues the kernel execution and also returns immediately. Then another `cudaMemcpyAsync()` enqueues the device-to-host (D→H) result transfer, again returning immediately. Finally, `cudaStreamSynchronize()` blocks until all enqueued operations complete. From the CPU perspective, the three middle operations return immediately after enqueueing work to the CUDA stream—they do not wait for execution. The CPU thread is free to prepare the next batch of work, perform other computations, or handle I/O while the GPU processes the current batch. This asynchronous behavior is crucial for performance, as it maximizes hardware utilization by keeping both CPU and GPU busy simultaneously.

However, this asynchronous design makes the actual GPU execution timeline completely opaque to CPU-side tools. The GPU timeline shows that after the blocking `cudaMalloc()` allocates device memory (~1μs), all the enqueued operations execute sequentially in the stream in the order they were submitted. First, the **H→D PCIe DMA transfer** (200 microseconds) moves data across the PCIe bus from host memory to GPU device memory. This transfer uses dedicated copy engines, so its duration depends on PCIe bandwidth (Gen3: ~12 GB/s, Gen4: ~25 GB/s), transfer size, and whether the host memory is pinned (page-locked). Critically, the CPU has already moved on to other work after the `cudaMemcpyAsync()` call returned, completely unaware of when this transfer actually starts, how long it takes, or when it completes.

Following the H→D memory transfer, there is a brief **kernel launch and setup overhead** (around 5 microseconds in this example) where the GPU runtime configures the kernel for execution—validating parameters, setting up grid/block configuration, and scheduling on the hardware queue. This overhead occurs entirely on the GPU side and is invisible to the CPU. Then comes the actual **kernel execution phase** (100 microseconds shown here), where SMs execute the parallel computation. During this phase, critical performance characteristics emerge: warps may experience divergence when threads take different control flow paths, memory accesses may suffer from poor coalescing or bank conflicts, and SM occupancy may fluctuate due to resource constraints—but none of these behaviors are visible to traditional tools. Finally, the **device-to-host DMA transfer** (150 microseconds) copies results back to host memory, again using dedicated copy engines. This D→H transfer was enqueued by the second `cudaMemcpyAsync()` call but only begins executing after the kernel completes. Throughout all of this GPU-side activity (455 microseconds total), the CPU remains oblivious to the actual transfer timing, computation progress, and whether operations are bottlenecked.

The key insight from this timeline is that the complete workflow (allocate → H→D transfer → compute → D→H transfer) consists of multiple distinct sequential phases on the GPU, each with different performance characteristics and optimization opportunities, but they execute while the CPU is doing other work. The asynchronous design decouples CPU-side enqueueing from GPU-side execution, maximizing throughput but obscuring visibility. For example, if the H→D memory transfer is slow due to unpinned memory (e.g., takes 400μs instead of 200μs), it delays all subsequent operations including kernel execution by 200μs, even if the kernel itself is well-optimized—but this bottleneck is invisible to CPU-side tools. If the kernel finishes quickly but has poor SM utilization, precious compute resources sit idle during the execution phase. If the D→H transfer is unnecessarily large (e.g., copying back more data than needed), it wastes PCIe bandwidth and delays when the CPU can start processing results. Traditional CPU-side tracing tools cannot measure these individual phases of the workflow, cannot determine which phase is the bottleneck (memory allocation, H→D transfer, launch overhead, computation, or D→H transfer), and cannot detect inefficiencies within each phase—they only see the total elapsed time from the initial `cudaMalloc()` to the final `cudaStreamSynchronize()`, obscuring where optimization efforts should focus within the allocate→transfer→compute→transfer pipeline.

#### What Traditional Tools Miss

This visibility gap means traditional CPU-side tracing only sees API calls and synchronization points, missing critical GPU-internal behaviors:

- **Launch overhead**: Each kernel launch incurs 5-50 microseconds of setup overhead on the GPU side. For synchronous kernels, this overhead is conflated with execution time. For asynchronous kernels, it is completely invisible since the launch API returns immediately. This makes it impossible to detect when applications launch too many small kernels that would benefit from batching or kernel fusion.

- **Memory transfer timing**: CPU-side tools see when `cudaMemcpy()` or `cudaMemcpyAsync()` is called, but not when the actual PCIe DMA transfer occurs, how long it takes, or whether it overlaps with computation. This hides critical optimization opportunities like using streams to overlap transfers with kernels, or switching to pinned memory to accelerate transfers.

- **Execution gaps and SM underutilization**: Even when a kernel is "running" from the CPU perspective, the GPU may be underutilized due to insufficient parallelism (too few blocks/threads), resource constraints (register pressure or shared memory limits reducing occupancy), or load imbalance (some SMs finishing early while others are still busy). These gaps represent wasted compute capacity but are invisible without GPU-side instrumentation.

- **Warp-level behavior**: The most critical performance characteristics—**thread divergence** (when threads in a warp take different control flow paths, causing serialization), **memory coalescing failures** (when threads access non-contiguous memory addresses, causing multiple transactions instead of one), and **shared memory bank conflicts** (when threads simultaneously access the same memory bank, causing serialization)—all occur at the warp and instruction level during kernel execution. Traditional tools cannot observe these behaviors, making it nearly impossible to diagnose why a kernel is slow or predict the impact of code changes.

This observability gap is precisely what bpftime's GPU support addresses. By enabling eBPF programs to run natively inside GPU kernels, bpftime provides nanosecond-granularity visibility into every phase of GPU execution: developers can measure actual kernel execution time separate from launch overhead, correlate memory access patterns with performance degradation, detect warp divergence and bank conflicts as they occur, and adaptively collect telemetry based on runtime conditions—all without recompiling the application or imposing the 10-100x overhead typical of traditional GPU profilers.

**bpftime's approach** bridges this gap by extending eBPF's programmability and customization model directly into GPU execution contexts, enabling eBPF programs to run natively inside GPU kernels alongside application workloads. The system defines a comprehensive set of GPU-side attach points that mirror the flexibility of CPU-side kprobes/uprobes: developers can instrument CUDA/ROCm device function entry and exit points (analogous to function probes), thread block lifecycle events (block begin/end), synchronization primitives (barriers, atomics), memory operations (loads, stores, transfers), and stream/event operations. eBPF programs written in restricted C are compiled through LLVM into device-native bytecode—PTX (Parallel Thread Execution) assembly for NVIDIA GPUs or SPIR-V for AMD/Intel—and dynamically injected into target kernels at runtime through binary instrumentation, without requiring source code modification or recompilation. The runtime provides a full eBPF execution environment on the GPU including: (1) a safety verifier to ensure bounded execution and memory safety in the SIMT context, (2) a rich set of GPU-aware helper functions for accessing thread/block/grid context, timing, synchronization, and formatted output, (3) specialized BPF map types that live in GPU memory for high-throughput per-thread data collection (GPU array maps) and event streaming (GPU ringbuf maps), and (4) a host-GPU communication protocol using shared memory and spinlocks for safely calling host-side helpers when needed. This architecture enables not only collecting fine-grained telemetry (per-warp timing, memory access patterns, control flow divergence) at nanosecond granularity, but also adaptively modifying kernel behavior based on runtime conditions, building custom extensions and optimizations, and unifying GPU observability with existing CPU-side eBPF programs into a single analysis pipeline—all while maintaining production-ready overhead characteristics. This enables:

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
