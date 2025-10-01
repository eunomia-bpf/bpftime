# threadhist - GPU Thread Execution Histogram

## Overview

When you launch a CUDA kernel with multiple threads, you expect each thread to do roughly equal work. But what if thread 0 executes the kernel 200,000 times while thread 4 only runs 150,000 times? That's a 25% workload imbalance that's silently degrading your GPU performance.

`threadhist` reveals these hidden load imbalances by counting how many times each GPU thread actually executes your kernel. Unlike traditional profilers that show aggregate metrics, this tool exposes per-thread execution patterns that directly impact performance.

## Understanding GPU Thread Execution

When you launch a CUDA kernel like `vectorAdd<<<1, 5>>>()`, you're creating 5 threads (threadIdx.x = 0 through 4) that should process your data in parallel. In an ideal world, all 5 threads would execute the kernel the same number of times and do equal work.

However, several factors can cause **load imbalance**:

- **Grid-stride loops**: The last thread might process fewer elements due to array size not dividing evenly
- **Conditional branches**: Some threads might skip work based on their index or data values
- **Early exits**: Threads might return early when they hit boundary conditions
- **Poor work distribution**: The algorithm itself might assign unequal work to different threads

These imbalances mean some threads are busy while others are idle, wasting precious GPU compute capacity. But because they happen inside the kernel, traditional profiling tools can't see them—they only report that "the kernel took X milliseconds."

## What This Tool Captures

`threadhist` uses a GPU array map to maintain a per-thread counter. Every time a thread exits the kernel, it increments its counter by 1. The userspace program periodically reads these counters and displays a histogram showing exactly how many times each thread has executed.

## Why This Matters: Real Performance Stories

### The Grid-Stride Workload Imbalance

You're processing a 1 million element array with a grid-stride loop. Your kernel launches with 5 threads, and you expect each to process about 200,000 elements. After running `threadhist` for a few seconds, you see:

```
Thread 0: 210432
Thread 1: 210432
Thread 2: 210432
Thread 3: 210432
Thread 4: 158304  (Only 75% of the work!)
```

What's happening? Your kernel is being invoked repeatedly in a loop, and each invocation processes a chunk of data. Due to how your grid-stride loop is written, thread 4 consistently finishes early because the remaining elements divide unevenly. While threads 0-3 continue working, thread 4 sits idle.

**The fix**: Adjust your thread block configuration or restructure the grid-stride loop to distribute the boundary work more evenly. After optimization, all threads show similar counts, indicating balanced workload.

### The Conditional Branch Mystery

You're running a kernel that processes data with some conditional logic. The histogram reveals:

```
Thread 0: 195423
Thread 1: 195423
Thread 2: 98156   (50% fewer executions!)
Thread 3: 195423
Thread 4: 195423
```

Thread 2 is executing significantly less often than the others. Looking at your code, you discover there's a conditional that causes thread 2 to exit early in certain cases:

```cuda
if (threadIdx.x == 2 && someCondition()) {
    return;  // Early exit
}
```

This pattern means thread 2 is doing half the work, but the other threads in the warp have to wait for it during the iterations where it does execute. This is **warp divergence** causing serialization, and the idle time from thread 2's early exits wastes GPU cycles.

**The insight**: Either remove the branch to make all threads follow the same path, or restructure your data so this condition doesn't correlate with specific thread indices.

### Detecting Completely Idle Threads

In a more extreme case, you might see:

```
Thread 0: 187234
Thread 1: 187234
Thread 2: 187234
Thread 3: 0       (Never executed!)
Thread 4: 0       (Never executed!)
```

Threads 3 and 4 aren't executing at all! This indicates a bug in your kernel launch configuration or grid-stride logic. Perhaps your workload size only requires 3 threads, but you're launching 5—wasting GPU resources. Or maybe there's a bug where certain thread indices never enter the main processing loop.

**The action**: Adjust your kernel launch parameters to match actual workload requirements, or fix the loop logic to ensure all threads participate.

## Building

```bash
# From bpftime root directory
make -C example/gpu/threadhist
```

Requirements:
- bpftime built with CUDA support (`-DBPFTIME_ENABLE_CUDA_ATTACH=1`)
- CUDA toolkit installed

## Running

### Terminal 1: Start the histogram collector
```bash
BPFTIME_LOG_OUTPUT=console LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  example/gpu/threadhist/threadhist
```

### Terminal 2: Run your CUDA application
```bash
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so  example/gpu/threadhist/vec_add
```

Or trace any CUDA application:
```bash
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  ./your_cuda_app
```

## Example Output

```
12:34:56
Thread 0: 210432
Thread 1: 210432
Thread 2: 210432
Thread 3: 210432
Thread 4: 158304
```

The timestamp shows when the snapshot was taken, followed by the total execution count for each thread since the program started.

## Use Cases

### 1. Optimizing Thread Block Configuration

You're experimenting with different block sizes. By running `threadhist` with various configurations, you can quickly see which configuration produces the most balanced workload distribution, maximizing GPU utilization.

### 2. Validating Grid-Stride Loop Implementations

After implementing or modifying a grid-stride loop, verify that all threads are executing roughly equally. Large discrepancies indicate the loop isn't distributing work evenly.

### 3. Detecting Algorithmic Imbalances

Some algorithms inherently create load imbalance (e.g., processing a sparse matrix where some threads have many elements, others few). The histogram quantifies this imbalance, helping you decide whether to redesign the algorithm or accept the tradeoff.

### 4. Debugging Thread Launch Issues

If threads show zero executions, you've caught a bug in your launch configuration or kernel logic before it becomes a production issue.

## How It Works

1. Attaches an eBPF kretprobe to the target CUDA kernel function
2. On kernel exit, each GPU thread increments its counter in the GPU array map: `*cnt += 1`
3. The GPU array map allocates per-thread storage automatically (one `u64` per thread)
4. Userspace periodically reads the entire array and prints the histogram
5. Counters accumulate over time, showing total executions since program start

## Code Structure

- **`threadhist.bpf.c`**: eBPF program running on GPU at kernel exit, incrementing per-thread counters
- **`threadhist.c`**: Userspace loader that reads and displays the histogram
- **`vec_add.cu`**: Example CUDA application for testing

## Limitations

- Shows cumulative counts since program start (not per-second rates)
- Fixed thread count (hardcoded to 7 threads in the example, line 87 of `threadhist.c`)
- Only tracks kernel exits (doesn't show per-invocation timing)
- Requires knowing the kernel function symbol name

## Customization

To trace a different kernel, modify the SEC annotation in `threadhist.bpf.c`:

```c
SEC("kretprobe/_Z9vectorAddPKfS0_Pf")  // Current: vectorAdd(const float*, const float*, float*)
```

Find your kernel's mangled name with:
```bash
cuobjdump -symbols your_app | grep your_kernel_name
```

To monitor more threads, change the thread count parameter in `threadhist.c:87`:
```c
print_stat(skel, 32);  // Monitor 32 threads instead of 7
```

## Interpreting Results

**Perfectly balanced** (all threads ±5%):
```
Thread 0: 200000
Thread 1: 199876
Thread 2: 200124
```
✓ Excellent - GPU resources fully utilized

**Slight imbalance** (10-20% variance):
```
Thread 0: 200000
Thread 1: 180000
```
⚠ Acceptable for complex algorithms, but investigate if possible

**Severe imbalance** (>25% variance):
```
Thread 0: 200000
Thread 4: 120000
```
❌ Performance problem - restructure workload distribution

**Idle threads** (zero counts):
```
Thread 3: 0
```
❌ Bug or misconfiguration - fix immediately

## Troubleshooting

**All threads show zero**: eBPF program didn't attach. Check kernel name matches exactly (including C++ mangling).

**Counts don't match expectations**: Ensure you're measuring the correct kernel. Use `cuobjdump` to verify the symbol name.

**Output doesn't update**: The application might not be invoking the kernel. Check that both processes are running and communicating via shared memory.
