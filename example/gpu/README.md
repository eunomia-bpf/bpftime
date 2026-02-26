# Write and Run eBPF on GPU with bpftime

> Yusheng Zheng=, Tong Yu=, Yiwei Yang=

bpftime provides GPU support through its CUDA/ROCm attachment implementation, enabling eBPF programs to execute **within GPU kernels** on NVIDIA and AMD GPUs. This brings eBPF's programmability, observability, and customization capabilities to GPU computing workloads, enabling real-time profiling, debugging, and runtime extension of GPU applications without source code modification.

> **Note:** GPU support is still experimental. For questions or suggestions, [open an issue](https://github.com/eunomia-bpf/bpftime/issues) or [contact us](mailto:team@eunomia.dev).

## The Problem: GPU Observability Challenges

GPUs have become the dominant accelerators for machine learning, scientific computing, and high-performance computing workloads, but their SIMT (Single Instruction, Multiple Thread) execution model introduces significant observability and extensibility challenges. Modern GPUs organize thousands of threads into warps (typically 32 threads) that execute in lockstep on streaming multiprocessors (SMs), with kernels launched asynchronously from the host. These threads navigate complex multi-level memory hierarchies ranging from fast but limited per-thread registers, to shared memory/LDS within thread blocks, through L1/L2 caches, to slower but abundant device memory, while contending with limited preemption capabilities that make kernel execution difficult to interrupt, inspect, or extend. This architectural complexity creates rich performance characteristics including warp divergence, memory coalescing patterns, bank conflicts, and occupancy variations that directly impact throughput, yet these behaviors remain largely opaque to traditional observability tools. Understanding and optimizing issues like kernel stalls, memory bottlenecks, inefficient synchronization, or suboptimal SM utilization requires fine-grained visibility into the execution flow, memory access patterns, and inter-warp coordination happening deep inside the GPU, along with the ability to dynamically inject custom logic. These capabilities are what existing tooling struggles to provide in a flexible, programmable manner.

Existing GPU tracing and profiling tools fall into two categories, each with significant limitations. First, many tracing tools operate exclusively at the CPU-GPU boundary by intercepting CUDA/ROCm userspace library calls (e.g., via LD_PRELOAD hooks on libcuda.so) or instrumenting kernel drivers at the system call layer. While this approach captures host-side events like kernel launches, memory transfers, and API timing, it fundamentally treats the GPU as a black box, providing no visibility into what happens during kernel execution, no correlation with specific warp behaviors or memory stalls, and no ability to adaptively modify behavior based on runtime conditions inside the device. Second, GPU vendor-specific profilers (NVIDIA's CUPTI, Nsight Compute, Intel's GTPin, AMD's ROCProfiler, research tools like NVBit or Neutrino) do provide device-side instrumentation and can collect hardware performance counters, warp-level traces, or instruction-level metrics. However, these tools operate in isolated ecosystems disconnected from Linux kernel observability and extension stacks. They cannot correlate GPU events with CPU-side eBPF probes (kprobes, uprobes, tracepoints), require separate data collection pipelines and analysis workflows, often impose substantial overhead (10-100x slowdowns for fine-grained instrumentation), and lack the dynamic programmability and control from the control plane that makes eBPF powerful for customizing what data to collect and how to process it in production environments, without recompilation or service interruption.


### The Timeline Visibility Gap: What Can and Cannot Be Observed

Consider a common debugging scenario. "My CUDA application takes 500ms to complete, but I don't know where the time is spent. Is it memory transfers, kernel execution, or API overhead?" The answer depends critically on whether the application uses synchronous or asynchronous CUDA APIs, and reveals fundamental limitations in CPU-side observability.

#### Synchronous Execution: What CPU-Side Tools Can and Cannot See

In synchronous mode, CUDA API calls block until the GPU completes each operation, creating a tight coupling between CPU and GPU timelines. Consider a typical workflow of allocate device memory, transfer data to GPU (host-to-device), execute kernel, and wait for completion. CPU-side profilers can measure the wall-clock time of each blocking API call, providing useful high-level insights. For example, if `cudaMemcpy()` takes 200μs while `cudaDeviceSynchronize()` (which waits for kernel completion) takes only 115μs, a developer can quickly identify that data transfer dominates over computation, suggesting a PCIe bottleneck that might be addressed by using pinned memory, larger batch sizes, or asynchronous transfers.

```
CPU Timeline (what traditional tools see):
───────────────────────────────────────────────────────────────────────────
 cudaMalloc()  cudaMemcpy()       cudaLaunchKernel()  cudaDeviceSync()
──────●─────────●──────────────────●──────────────────●────────────────────
   ~1μs wait     200μs wait        returns           115μs wait
   (blocked)     (H→D transfer)    immediately       (kernel done)

GPU Timeline (actual execution with hidden phases):
───────────────────────────────────────────────────────────────────────────
      ◄─Alloc─►◄────H→D DMA────►◄──Launch──►◄──Kernel Exec──►◄─Cleanup─►
      │ ~1μs  │     200μs        │   5μs    │     100μs       │  ~10μs  │
──────┴───────┴──────────────────┴──────────┴─────────────────┴─────────┴──
                                           (SM busy)                (SM idle)
```

However, when the developer asks "The kernel sync takes 115μs, but why is my kernel slow? Is it launch overhead, memory stalls, warp divergence, or low SM utilization?", CPU-side tools hit a fundamental wall. The 115μs sync time is an opaque aggregate that conflates multiple hidden GPU-side phases including kernel launch overhead (~5μs to schedule work on SMs), actual kernel execution (~100μs of computation on streaming multiprocessors), and cleanup (~10μs to drain pipelines and release resources), as shown in the GPU timeline above.

Even with perfect timing of synchronous API calls, CPU-side tools cannot distinguish whether poor kernel performance stems from (1) excessive launch overhead (e.g., too many small kernel launches), (2) compute inefficiency within the 100μs execution window (e.g., only 30% of warps are active due to divergence), (3) memory access patterns causing stalls (e.g., uncoalesced global memory loads), or (4) SM underutilization (e.g., only 50% of available SMs are busy). These require visibility into warp-level execution, memory transaction statistics, and per-thread behavior that is only accessible from inside the GPU during kernel execution. We need a finer-grained, programable, GPU-internal perspective to understand what is happening during kernel execution, not just when it starts and ends.

#### Asynchronous Execution: How Temporal Decoupling Eliminates Visibility

Modern CUDA applications use asynchronous APIs (`cudaMemcpyAsync()`, `cudaLaunchKernel()` with streams) to maximize hardware utilization by overlapping CPU work with GPU execution. This introduces temporal decoupling where API calls return immediately after enqueuing work to a stream, allowing the CPU to continue executing while the GPU processes operations sequentially in the background. This breaks the observability that CPU-side tools had in synchronous mode.

Consider the same workflow now executed asynchronously. The developer enqueues a host-to-device transfer (200μs), kernel launch (100μs execution), and device-to-host transfer (150μs), then continues CPU work before eventually calling `cudaStreamSynchronize()` to wait for all GPU operations to complete. From the CPU's perspective, all enqueue operations return in microseconds, and only the final sync point blocks, reporting a total of 455μs (200 + 5 + 100 + 150 μs of sequential GPU work).

```
CPU Timeline (what traditional tools see):
─────────────────────────────────────────────────────────────────────────────────
cudaMallocAsync() cudaMemcpyAsync() cudaLaunchKernel() cudaMemcpyAsync()      cudaStreamSync()
●─────●─────●─────●─────────────────────────────────────────────────────────────────●────
1μs  1μs  1μs  1μs         CPU continues doing other work...               455μs wait
(alloc)(H→D)(kernel)(D→H)                                                       (all done)

GPU Timeline (actual execution - sequential in stream):
─────────────────────────────────────────────────────────────────────────────────
◄─Alloc─►◄───────H→D DMA────────►◄─Launch─►◄────Kernel Exec────►◄────D→H DMA────►
│ ~1μs  │       200μs            │   5μs   │       100μs        │      150μs     │
┴────────┴────────────────────────┴─────────┴────────────────────┴────────────────┴─────
         ↑                                  ↑                                     ↑
    CPU already moved on              GPU still working                    Sync returns
```

In synchronous execution, measuring individual API call durations allowed developers to identify whether transfers or compute dominated. In asynchronous mode, this capability disappears entirely as all timing information is collapsed into a single 455μs aggregate at the sync point. The question "Is my bottleneck memory transfer or kernel execution?" becomes unanswerable from the CPU side. If the first transfer takes twice as long due to unpinned memory (400μs instead of 200μs), delaying all subsequent operations by 200μs, the developer only sees the total time increase from 455μs to 655μs with zero indication of which operation caused the delay, when it occurred, or whether it propagated to downstream operations.

Asynchronous execution not only hides the GPU-internal details that were already invisible in synchronous mode (warp divergence, memory stalls, SM utilization), but also eliminates the coarse-grained phase timing that CPU tools could previously provide. Developers lose the ability to even perform basic triage. They cannot determine whether to focus optimization efforts on memory transfers, kernel logic, or API usage patterns without either (1) reverting to slow synchronous execution for debugging (defeating the purpose of async), or (2) adding manual instrumentation that requires recompilation and provides only static measurement points.

Modern GPU applications like LLM serving further complicate this picture with advanced optimization techniques. Batching strategies combine multiple operations to maximize throughput like a pipeline, but make it harder to identify which individual operations are slow. Persistent kernels stay resident on the GPU processing multiple work batches, eliminating launch overhead but obscuring phase boundaries. Multi-stream execution with complex dependencies between streams creates intricate execution graphs where operations from different streams interleave unpredictably. Shared memory usage per thread block constrains occupancy and limits concurrent warp execution, creating subtle resource contention that varies based on kernel configuration. These optimizations significantly improve throughput but make the already-opaque async execution model even more difficult to observe and debug from the CPU side.

However, even if we had perfect GPU-internal visibility showing high memory stall cycles, we still couldn't determine the root cause: Is it warp divergence causing uncoalesced access? Host thread descheduling delaying async memory copies? PCIe congestion from concurrent RDMA operations? Or on-demand page migration latency from the kernel? Similarly, when observing SM underutilization, device-only metrics cannot distinguish whether the grid is too small, launches are serialized by a userspace mutex, or the driver throttled after an ECC error or power event.

The challenge becomes acute in production environments where tail latency spikes occur intermittently. Are they caused by GPU cache effects, cgroup throttling on the host producer thread, or interference from another container issuing large DMA transfers? Device-only tooling reports "what happened on the GPU," but not "why it happened now in this heterogeneous system." Without time-aligned correlation to userspace behavior (threads, syscalls, allocations), kernel events (page faults, scheduler context switches, block I/O), and driver decisions (stream dependency resolution, memory registration, power state transitions), engineers must iterate slowly: observe a GPU symptom → guess a host cause → rebuild with ad-hoc instrumentation → repeat. This feedback loop is expensive and often infeasible in latency-sensitive deployments.

> **The key insight:** Effective GPU observability and extensibility requires a unified solution that spans multiple layers of the heterogeneous computing stack: from userspace applications making CUDA API calls, through OS kernel drivers managing device resources, down to device code executing on GPU hardware. Traditional tools are fragmented across these layers, providing isolated visibility at the CPU-GPU boundary or within GPU kernels alone, but lacking the cross-layer correlation needed to understand how decisions and events at one level impact performance and behavior at another.

#### Current Tools: Nsight Systems/Compute and Their Limitations

NVIDIA's Nsight suite attempts to address some cross-domain visibility challenges. **Nsight Systems** provides system-wide timelines showing CPU threads, CUDA API calls, kernel launches, and memory transfers in a unified view. **Nsight Compute** offers deep kernel-level microarchitectural metrics via CUPTI counters and replay-based analysis, with guided optimization rules. These tools can correlate launches with CPU thread scheduling and provide rich per-kernel stall reasons and memory metrics.

However, Nsight and similar vendor tools suffer from fundamental limitations compared to a unified eBPF approach. First, Nsight offers a closed event model with a fixed set of events and no arbitrary programmable logic at attach points—you cannot dynamically add custom instrumentation without recompiling applications or write filtering predicates like "only collect data when kernel execution exceeds 100ms." Second, these tools require special profiling sessions that perturb workload behavior through counter multiplexing and replay mechanisms, making them unsuitable for always-on continuous telemetry in production, and causing replay-based collection to miss transient anomalies and rare events. Third, Nsight lacks in-situ filtering, forcing all raw data to be exported then post-processed, which creates multi-GB traces from large async pipelines with no programmable adaptive response to change sampling logic on-the-fly based on observed state. Fourth, system integration is severely limited: Nsight cannot attach dynamic probes to persistent kernels without restart, lacks integration with Linux eBPF infrastructure (kprobes/uprobes/tracepoints), and cannot share data structures (maps) across CPU and GPU instrumentation, making it extremely difficult to stitch causality chains like page fault (host) → delayed launch enqueue → warp stall spike. Finally, these are NVIDIA-only tools with no clear path to vendor-neutral deployment across AMD, Intel, or other accelerators in heterogeneous systems. In practice, developers face iterative root cause analysis slowdowns with large traces, miss production issues that don't reproduce under profiling overhead, and cannot correlate GPU events with existing production observability stacks (perf, bpftrace, custom eBPF agents) without complex mode-switching to special "profiling sessions."

#### Bridging the Gap: Unified CPU and GPU Observability and Extensibility with eBPF

By running eBPF programs natively inside GPU kernels, bpftime provides programmable, unified observability and extensibility across the entire stack. Unlike session-based profilers, it enables always-on production monitoring with dynamic load/unload of probes and device-side predicate filtering to reduce overhead. It recovers async-mode visibility with per-phase timestamps (H→D at T+200μs, kernel at T+205μs, D→H at T+455μs), exposes GPU-internal details with nanosecond-granularity telemetry for warp execution and memory patterns, and correlates CPU and GPU events without the heavyweight overhead of traditional separate profilers.

The architecture treats CPU and GPU probes as peers in a unified control plane: shared BPF maps and ring buffers enable direct data exchange, dynamic instrumentation works without recompilation or restart, and integration with existing eBPF infrastructure (perf, bpftrace, custom agents) requires no mode-switching. Developers can simultaneously trace CPU-side CUDA API calls via uprobes, kernel driver interactions via kprobes, and GPU-side kernel execution via CUDA probes, all using the same eBPF toolchain and correlating events across the host-device boundary. Example questions now become answerable: "Did the CPU syscall delay at T+50μs cause the GPU kernel to stall at T+150μs?" or "Which CPU threads are launching the kernels that exhibit high warp divergence?" This cross-layer visibility enables root-cause analysis that spans the entire heterogeneous execution stack, from userspace application logic through kernel drivers to GPU hardware behavior, without leaving the production observability workflow.

## The Solution: eBPF on GPU with bpftime

**bpftime's approach** bridges this gap by extending eBPF's programmability and customization model directly into GPU execution contexts, enabling eBPF programs to run natively inside GPU kernels alongside application workloads. The system defines a comprehensive set of GPU-side attach points that mirror the flexibility of CPU-side kprobes/uprobes. Developers can instrument CUDA/ROCm device function entry and exit points (analogous to function probes), thread block lifecycle events (block begin/end), synchronization primitives (barriers, atomics), memory operations (loads, stores, transfers), and stream/event operations. eBPF programs written in restricted C are compiled through LLVM into device-native bytecode (PTX (Parallel Thread Execution) assembly for NVIDIA GPUs or SPIR-V for AMD/Intel) and dynamically injected into target kernels at runtime through binary instrumentation, without requiring source code modification or recompilation. The runtime provides a full eBPF execution environment on the GPU including (1) a safety verifier to ensure bounded execution and memory safety in the SIMT context, (2) a rich set of GPU-aware helper functions for accessing thread/block/grid context, timing, synchronization, and formatted output, (3) specialized BPF map types that live in GPU memory for high-throughput per-thread data collection (GPU array maps) and event streaming (GPU ringbuf maps), and (4) a host-GPU communication protocol using shared memory and spinlocks for safely calling host-side helpers when needed. This architecture enables not only collecting fine-grained telemetry (per-warp timing, memory access patterns, control flow divergence) at nanosecond granularity, but also adaptively modifying kernel behavior based on runtime conditions, building custom extensions and optimizations, and unifying GPU observability with existing CPU-side eBPF programs into a single analysis pipeline, all while maintaining production-ready overhead characteristics. This enables:

- **3-10x faster performance** than tools like NVBit for instrumentation
- **Vendor-neutral design** that works across NVIDIA, AMD and Intel GPUs
- **Unified observability and control** with Linux kernel eBPF programs (kprobes, uprobes)
- **Fine-grained profiling and runtime customization** at the warp or instruction level
- **Adaptive GPU kernel memory optimization** and programmable scheduling across SMs
- **Dynamic extensions** for GPU workloads without recompilation
- **Accelerated eBPF applications** by leveraging GPU compute power

The architecture is designed to achieve four core goals: (1) provide a unified eBPF-based interface that works seamlessly across userspace, kernel, multiple CPU and GPU contexts from different vendors, (2) enable dynamic, runtime instrumentation without requiring source code modification or recompilation, and (3) maintain safe and efficient execution within the constraints of GPU hardware and SIMT execution models. (4) Less dependency and easy to deploy, built on top of existing CUDA/ROCm/OpenGL runtimes without requiring custom kernel drivers, firmware modifications, or heavy-weight runtimes like record-and-replay systems.

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

We demostarate GPU eBPF capabilities through several example, bcc style tools:

### [kernelretsnoop](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/kernelretsnoop) - Per-Thread Exit Timestamp Tracer

Attaches to CUDA kernel exits and records the exact nanosecond timestamp when each GPU thread completes execution. This reveals thread divergence, memory access patterns, and warp scheduling issues that are invisible to traditional profilers.

> `kprobe` / `kretprobe` here is a bad name, we just use it as a placeholder. I think it needs changed.

**Use case**: You notice your kernel is slower than expected. `kernelretsnoop` reveals that thread 31 in each warp finishes 750ns later than threads 0-30, exposing a boundary condition causing divergence. You refactor to eliminate the branch, and all threads now complete within nanoseconds of each other.

```c
// eBPF program runs on GPU at kernel exit
SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int ret__cuda() {
    u64 tid_x, tid_y, tid_z;
    bpf_get_thread_idx(&tid_x, &tid_y, &tid_z);  // Which thread am I?
    u64 ts = bpf_get_globaltimer();               // When did I finish?

    // Write to ringbuffer for userspace analysis
    bpf_perf_event_output(ctx, &events, 0, &data, sizeof(data));
}
```

### [threadhist](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/threadhist) - Thread Execution Count Histogram

Uses GPU array maps to count how many times each thread executes. Detects workload imbalance where some threads do far more work than others, wasting GPU compute capacity.

**Use case**: Your grid-stride loop processes 1M elements with 5 threads. You expect balanced work, but `threadhist` shows thread 4 executes only 75% as often as threads 0-3. The boundary elements divide unevenly, leaving thread 4 idle while others work. You adjust the distribution and achieve balanced execution.

```c
// eBPF program runs on GPU at kernel exit
SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int ret__cuda() {
    u64 tid_x, tid_y, tid_z;
    bpf_get_thread_idx(&tid_x, &tid_y, &tid_z);

    // Per-thread counter in GPU array map
    u64 *count = bpf_map_lookup_elem(&thread_counts, &tid_x);
    if (count) {
        __atomic_add_fetch(count, 1, __ATOMIC_SEQ_CST);  // Thread N executed once more
    }
}
```

### [launchlate](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/launchlate) - Kernel Launch Latency Profiler

Measures the time between `cudaLaunchKernel()` on CPU and actual kernel execution on GPU. Reveals hidden queue delays, stream dependencies, and scheduling overhead that make fast kernels slow in production.

**Use case**: Your kernels execute in 100μs each, but users report 50ms latency. `launchlate` shows 200-500μs launch latency per kernel because each waits for the previous one and memory transfers to complete. Total time is 5ms, not 1ms. You switch to CUDA graphs, batching all launches, and latency drops to 1.2ms.

```c
BPF_MAP_DEF(BPF_MAP_TYPE_ARRAY, launch_time);

// CPU-side uprobe captures launch time
SEC("uprobe/app:cudaLaunchKernel")
int uprobe_launch(struct pt_regs *ctx) {
    u64 ts_cpu = bpf_ktime_get_ns();  // When did CPU request launch?
    bpf_map_update_elem(&launch_time, &key, &ts_cpu, BPF_ANY);
}

// GPU-side kprobe captures execution start
SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int kprobe_exec() {
    u64 ts_gpu = bpf_get_globaltimer();  // When did GPU actually start?
    u64 *ts_cpu = bpf_map_lookup_elem(&launch_time, &key);

    u64 latency = ts_gpu - *ts_cpu;  // How long did kernel wait in queue?
    u32 bin = get_hist_bin(latency);
    // Update histogram...
}
```

### Other Examples

- **[cuda-counter](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/cuda-counter)**: Basic probe/retprobe with timing measurements
- **[mem_trace](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/mem_trace)**: Memory access pattern tracing and analysis
- **[directly_run_on_gpu](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/directly_run_on_gpu)**: Run eBPF programs directly on GPU without attaching to kernels
- **[rocm-counter](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/rocm-counter)**: AMD ROCm GPU instrumentation (experimental)

### Key Components

1. **CUDA Runtime Hooking**: Intercepts CUDA API calls using Frida-based dynamic instrumentation
2. **PTX Modification**: Converts eBPF bytecode to PTX (Parallel Thread Execution) assembly and injects it into GPU kernels
3. **Helper Trampoline**: Provides GPU-accessible helper functions for map operations, timing, and context access
4. **Host-GPU Communication**: Enables synchronous calls from GPU to host via pinned shared memory

### Attachment Types

bpftime supports three attachment types for GPU kernels (defined in `attach/nv_attach_impl/nv_attach_impl.hpp:33-34`):

- **`ATTACH_CUDA_PROBE` (8)** - Executes eBPF code at kernel entry
- **`ATTACH_CUDA_RETPROBE` (9)** - Executes eBPF code at kernel exit
- **Memory capture probe (`__memcapture`)** - Special probe type for capturing memory access patterns

All types support specifying target kernel functions by name (e.g., `_Z9vectorAddPKfS0_Pf` for mangled C++ names).

## GPU-Specific BPF Maps

**bpftime supports ALL maps on CPU, however, directly using CPU eBPF maps will significantly downgrade the performance.**

bpftime includes specialized map types optimized for GPU operations, that does not have much performance impact:

### `BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP` and `BPF_MAP_TYPE_GPU_ARRAY_MAP`

GPU-resident array maps with **per-thread storage** for high-performance data collection.

Key features:
- Data stored directly in GPU memory (CUDA IPC shared memory)
- Each thread gets isolated storage (`max_entries × max_thread_count × value_size`)
- Zero-copy access from GPU, DMA transfers to host
- Supports `bpf_map_lookup_elem()` and `bpf_map_update_elem()` in GPU code

Implementation: `runtime/src/bpf_map/gpu/nv_gpu_array_map.cpp:14-81`

### `BPF_MAP_TYPE_GPU_RINGBUF_MAP` (1527)

GPU ring buffer maps for efficient **per-thread event streaming** to host.

Key features:
- Lock-free per-thread ring buffers in GPU memory
- Variable-size event records with metadata
- Asynchronous data collection with low overhead
- Compatible with `bpf_perf_event_output()` helper

Implementation: `runtime/src/bpf_map/gpu/nv_gpu_ringbuf_map.cpp`

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
make build-gpu
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
