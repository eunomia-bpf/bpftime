# kernelretsnoop - CUDA Kernel Thread Exit Timestamp Tracer

## Note

Environment variable `BPFTIME_MAP_GPU_THREAD_COUNT` should be set to a smaller value (such as 10) to avoid bad_alloc.

## Overview

Imagine you're trying to understand why your CUDA kernel isn't as fast as you expected. Traditional profiling tools tell you the kernel took 10 milliseconds to run—but that's like knowing a marathon took 3 hours without knowing that half the runners got lost at mile 5. You need to see what each individual thread is doing, and when they're actually finishing their work.

`kernelretsnoop` gives you exactly that visibility. It's an eBPF-based tool that captures the precise moment each GPU thread exits your CUDA kernel, revealing timing patterns that are completely invisible to conventional profilers.

## Understanding GPU Threads

Before we dive into what `kernelretsnoop` does, let's understand what we're measuring. When you launch a CUDA kernel, you're not just running one piece of code, you're launching thousands or even millions of threads that execute in parallel on the GPU.

These threads are organized into a 3D grid structure. Each thread has coordinates (x, y, z) that identify its position in this grid. Threads are grouped into "warps" of 32 threads that execute together in lockstep. This means that ideally, all 32 threads in a warp should be doing the same thing at the same time. When they don't—when some threads take a different code path or wait on slow memory—the warp becomes less efficient, and that's where performance problems hide.

The challenge is that GPU hardware and traditional profilers only show you aggregate statistics: total kernel execution time, average occupancy, memory throughput. They can't tell you that thread 0 finished in 100 nanoseconds while thread 31 took 850 nanoseconds, or that threads processing the edges of your data consistently run slower than those in the middle.

## What This Tool Captures

`kernelretsnoop` attaches to the exit point of your CUDA kernel and, for each thread that completes, records:
- **Thread 3D coordinates** (x, y, z) - which specific thread this is within your kernel launch grid
- **GPU global timer timestamp** - the exact nanosecond when this thread finished executing

This simple data unlocks powerful insights into your kernel's behavior.

## Why This Matters

### The Case of the Divergent Warp

You launch a kernel with 1024 threads to process an array. When you run `kernelretsnoop`, you see something unexpected:

```
Thread (0, 0, 0) timestamp: 1749147474550023136
Thread (1, 0, 0) timestamp: 1749147474550023140  // +4ns
Thread (2, 0, 0) timestamp: 1749147474550023145  // +5ns
...
Thread (31, 0, 0) timestamp: 1749147474550023890 // +750ns later!
```

Threads 0 through 30 all finish within a few nanoseconds of each other—exactly what you'd expect from threads executing in lockstep. But thread 31 finishes 750 nanoseconds later. What's going on?

You check your kernel code and discover that thread 31 hits a boundary condition. it processes the last element of a chunk, triggering extra bounds checking or taking a different branch in your code. Because all 32 threads execute as a warp, when thread 31 takes this different path, it forces the entire warp to serialize: first executing the common path for threads 0-30, then executing the special path for thread 31. This is called **thread divergence**, and it's killing your performance.

Armed with this insight, you can refactor your code to eliminate the divergent branch, ensuring all threads in the warp follow the same path. After the change, all threads finish within nanoseconds of each other.

### The Memory Access Mystery

In another kernel, you notice a pattern when analyzing the timestamps:

```
Thread (0, 0, 0) timestamp: 1749147474550023140
Thread (8, 0, 0) timestamp: 1749147474550023890  // Much slower
Thread (16, 0, 0) timestamp: 1749147474550023150
Thread (24, 0, 0) timestamp: 1749147474550023900 // Much slower again
```

Every 8th thread takes significantly longer. This points to a **memory access pattern problem**. You realize that your data structure causes every 8th thread to access a different memory bank, creating bank conflicts, or worse, that these threads trigger cache misses while others hit the cache.

By correlating thread indices with timing, you've identified exactly which threads are experiencing memory bottlenecks. You can now restructure your data layout to ensure coalesced memory access, where consecutive threads access consecutive memory addresses—a pattern GPUs are optimized for.

### Understanding Warp Scheduling

If you aggressive the timestamps across multiple warps, you might discover that warps don't all execute in parallel:

```
Warp 0 (threads 0-31):   finish around timestamp 1749147474550023000
Warp 1 (threads 32-63):  finish around timestamp 1749147474550025000  // 2μs later
Warp 2 (threads 64-95):  finish around timestamp 1749147474550027000  // 4μs later
```

This reveals that your kernel is limited by resource constraints—perhaps register usage or shared memory—forcing warps to execute sequentially rather than in parallel. Now you know where to optimize: reduce register pressure or shared memory usage to increase parallelism.

## Building

```bash
# From bpftime root directory
make -C example/gpu/kernelretsnoop
```

Requirements:
- bpftime built with CUDA support (`-DBPFTIME_ENABLE_CUDA_ATTACH=1`)
- CUDA toolkit installed

## Running

### Terminal 1: Start the tracer
```bash
BPFTIME_SHM_MEMORY_MB=1000 BPFTIME_LOG_OUTPUT=console LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  example/gpu/kernelretsnoop/kernelretsnoop
```

### Terminal 2: Run your CUDA application
```bash
BPFTIME_LOG_OUTPUT=console LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  example/gpu/kernelretsnoop/vec_add
```

Or trace any CUDA application:
```bash
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  ./your_cuda_app
```

## Example Output

```
Thread (0, 0, 0) timestamp: 1749147474550023136
Thread (1, 0, 0) timestamp: 1749147474550023140
Thread (2, 0, 0) timestamp: 1749147474550023145
Thread (3, 0, 0) timestamp: 1749147474550023150
Thread (4, 0, 0) timestamp: 1749147474550023155
Total events collected: 5
```

## Use Cases

### 1. Diagnosing Performance Anomalies

You notice your kernel is slower than expected. `kernelretsnoop` reveals that 10% of threads take 5x longer to complete, pointing to a specific thread index pattern that accesses data differently.

### 2. Validating Optimizations

After rewriting memory access patterns, verify that timestamp deltas between threads decreased, confirming all threads now complete within nanoseconds of each other.

### 3. Understanding Boundary Conditions

Discover that threads processing array boundaries (e.g., thread indices near N) take longer due to additional boundary checks or unaligned access.

### 4. Debugging Race Conditions

Timestamp ordering reveals if assumptions about thread execution order are violated, helping identify synchronization bugs.

## How It Works

1. Attaches an eBPF kretprobe to the target CUDA kernel function (e.g., `vectorAdd`)
2. On kernel exit, each GPU thread executes the eBPF code
3. The eBPF program calls GPU-specific helpers:
   - `bpf_get_thread_idx()` - Current thread coordinates
   - `bpf_get_globaltimer()` - GPU nanosecond-precision timer
4. Data is written to a GPU ringbuffer
5. Userspace polls and displays the results

## Code Structure

- **`kernelretsnoop.bpf.c`**: eBPF program running on GPU at kernel exit
- **`kernelretsnoop.c`**: Userspace loader and output handler
- **`vec_add.cu`**: Example CUDA application for testing

## Limitations

- Only captures kernel **exit** timestamps (not entry)
- Does not capture kernel arguments or return values
- Ringbuffer overhead may affect absolute timing (but relative timing between threads remains accurate)
- Requires kernel function symbol name (mangled C++ name)

## Customization

To trace a different kernel, modify the SEC annotation in `kernelretsnoop.bpf.c`:

```c
SEC("kretprobe/_Z9vectorAddPKfS0_Pf")  // Current: vectorAdd(const float*, const float*, float*)
```

Find your kernel's mangled name with:
```bash
cuobjdump -symbols your_app | grep your_kernel_name
```

## Advanced Analysis

Export timestamps to a file for deeper analysis:

```bash
./kernelretsnoop | tee timestamps.txt
```

Then use scripts to:
- Calculate per-warp timing variance
- Identify thread index patterns with outliers
- Correlate timestamps with thread coordinates for access pattern visualization

## Troubleshooting

**No output**: Ensure the eBPF program attached successfully. Check that the kernel name matches exactly (including C++ mangling).

**Incomplete data**: Ringbuffer may be too small for high-frequency kernels. Increase `max_entries` in the ringbuf map definition.

**Timestamps out of order**: Normal if warps execute in parallel. Sort by timestamp if sequential ordering is needed for analysis.
