
# launchlate - CUDA Kernel Launch Latency Profiler

## Overview

Your GPU kernels execute in microseconds, but sometimes your application is mysteriously slow. Traditional profilers show kernel execution time, but they don't reveal how long kernels wait between being launched from CPU and actually starting execution on GPU. This "launch latency"—caused by stream dependencies, resource contention, or driver overhead—can add 10-100x more delay than the kernel itself takes to run, especially for short-lived kernels.

`launchlate` measures the time gap between `cudaLaunchKernel()` calls on the CPU and when kernels actually start executing on the GPU. It reveals the hidden queuing delays, scheduling overhead, and synchronization costs that make your fast kernels slow in production.

## Understanding Kernel Launch Latency

When you call `cudaLaunchKernel()` from CPU code, you're not directly starting GPU execution—you're queuing work. Here's what actually happens:

1. **CPU enqueues the kernel** into a CUDA stream (microseconds)
2. **Driver processes the command** and prepares kernel parameters
3. **Work scheduler waits** for resources and prior operations to complete
4. **GPU finally receives** the kernel and begins execution
5. **Streaming multiprocessors** start running your code

The time between steps 1 and 5 is your launch latency. In ideal conditions, this might be 5-20 microseconds. But in production systems with multiple streams, complex dependencies, memory pressure, or busy GPUs, this can balloon to milliseconds—longer than the kernel itself runs.

## What This Tool Measures

`launchlate` uses a dual-probe approach to capture both sides of the launch latency gap:

1. **CPU-side uprobe** on `cudaLaunchKernel()` - captures when kernels are launched from host code
2. **GPU-side kprobe** on the actual kernel function - captures when execution begins on device
3. **Time correlation** - calculates the delta between launch and execution, with clock calibration
4. **Histogram analysis** - categorizes latencies into bins from nanoseconds to seconds

The result is a real-time histogram showing the distribution of launch latencies, revealing patterns invisible to traditional profilers.

## Why This Matters

### Stream Dependency Overhead

Your inference pipeline runs multiple kernels in sequence. Each kernel executes in 100μs, so you expect 10 kernels to take 1ms total. But `launchlate` shows most kernels have 200-500μs launch latency because each kernel waits for the previous one to finish, memory transfers to complete, and the GPU scheduler to process the queue. The total time is actually 3-5ms, not 1ms. Solution: use CUDA graphs to batch launches, or persistent kernels to eliminate per-launch overhead.

### Small Kernel Launch Overhead

You've written clean, modular code with many small specialized kernels. Each kernel runs in 10-20μs, but `launchlate` reveals launch latency of 30-80μs per kernel—you're spending 3-5x more time launching kernels than executing them. This is the classic small-kernel trap: the fixed cost of kernel launch (driver processing, queue management, resource allocation) dominates when kernels are tiny. Fuse kernels together or batch work to amortize the launch cost.

### Tail Latency from Context Switching

Your GPU inference service runs smoothly with P50 launch latency of 25μs, but P99 spikes to 2ms causing timeout errors. `launchlate` histogram shows a small but consistent spike in the 1-10ms bin. The cause: your GPU is shared by multiple processes (monitoring, other services), and occasional CUDA context switches add milliseconds of overhead. Solutions: dedicate GPUs per service, use MPS (Multi-Process Service), or increase timeouts to accommodate tail latency.

### PCIe Contention

Your application does GPU inference while streaming data over a 100Gb NIC that shares the same PCIe root complex. Most kernels launch quickly, but `launchlate` shows periodic spikes to 500μs-2ms that correlate with network bursts. The GPU and NIC are competing for PCIe bandwidth, and when the NIC saturates the bus, GPU commands queue up waiting for PCIe access. Solution: spread devices across multiple PCIe root complexes, or schedule network I/O and GPU work to avoid overlap.

## Building the Example

```bash
# Navigate to the bpftime root directory
cd bpftime

# Build the main bpftime project first
make build-gpu

# Build the example
make -C example/gpu/launchlate-kernel-gpu-shared-map
```

## Running the Example

You need to start two processes:

### 1. Launch the eBPF Program (Server)

```bash
BPFTIME_LOG_OUTPUT=console BPFTIME_NOT_LOAD_PATTERN=cuda.*  BPFTIME_RUN_WITH_KERNEL=true bpftime load ./launchlate
```

This process loads the eBPF program and waits for CUDA events.

### 2. Run the CUDA Application (Client)

In another terminal:

```bash
BPFTIME_LOG_OUTPUT=console LD_PRELOAD=build/runtime/agent/libbpftime-agent.so  example/gpu/launchlate/vec_add
```

This runs the vector addition program with the bpftime agent, which connects to the first process for eBPF execution.

## Example Output

When running successfully, you'll see:

```
Clock calibration: REALTIME - MONOTONIC = 1625284901234 ns
  MONOTONIC: 3842.123456789
  REALTIME:  1625288743.357890890

Monitoring CUDA kernel launch latency... Hit Ctrl-C to end.

12:34:56 Launch Latency Distribution:
latency         : count    distribution
100ns-1us       : 45       |********
1-10us          : 234      |****************************************
10-100us        : 167      |*****************************
100us-1ms       : 89       |***************
1-10ms          : 12       |**
Total samples: 547
```

This shows:
- Clock calibration between CPU monotonic clock and GPU timer
- Real-time histogram of launch latencies
- Distribution visualization showing where most latencies fall

## Use Cases

- **Diagnosing tail latency**: Understand why P99 request latency is 10x higher than P50 by identifying when launch latency spikes occur
- **Optimizing kernel granularity**: See if you're spending more time launching kernels than executing them
- **Debugging async performance gaps**: Measure actual queue delays when kernels wait for preceding operations
- **Multi-tenant GPU troubleshooting**: Identify context switch overhead when multiple processes share GPUs
- **System-level bottlenecks**: Detect PCIe contention, driver CPU overhead, or resource allocation delays
- **Validating optimizations**: Quantify improvements from CUDA graphs, kernel fusion, or persistent kernels

## How It Works

1. **CPU-side uprobe** on `cudaLaunchKernel()` captures when the kernel is launched and records a timestamp
2. **GPU-side kprobe** on the kernel function captures when execution actually starts on the GPU
3. **Clock calibration** synchronizes CPU and GPU timers to enable accurate comparison
4. **Latency calculation** computes the time difference and updates a histogram showing the distribution
5. **Real-time display** shows the histogram every few seconds, revealing launch latency patterns

## Code Structure

- **`launchlate.bpf.c`**: eBPF programs that run on CPU (uprobe) and GPU (kprobe) to capture timestamps and compute latency histogram
- **`launchlate.c`**: Userspace program that loads eBPF code, calibrates clocks, and displays the histogram
- **`vec_add.cu`**: Example CUDA application that repeatedly launches kernels for testing

## Limitations

- Requires kernel function symbol names (use `cuobjdump -symbols your_app | grep kernel_name`)
- Clock drift may affect accuracy over long runs (recalibrate periodically if needed)
- Only tracks one kernel function at a time (attach multiple probes for multiple kernels)

## Customization

To trace your own kernel, change the kprobe target in `launchlate.bpf.c`:

```c
SEC("kprobe/_Z15yourKernelNamePfS_")  // Replace with your kernel's mangled name
```

Find your kernel's symbol name:
```bash
cuobjdump -symbols your_app | grep your_kernel_name
```

## Interpreting Results

**Good**: Most launches under 100μs
```
1-10us    : 450  |****************************************
10-100us  : 89   |********
```

**Warning**: Significant tail latency over 1ms
```
100us-1ms : 234  |********************
1-10ms    : 367  |********************************
```
Investigate stream dependencies or multi-tenancy issues

**Problem**: Many launches over 10ms
```
10-100ms  : 245  |****************************************
100ms-1s  : 123  |********************
```
Major bottleneck—check for context switches, PCIe contention, or driver issues

## Troubleshooting

**No output**: Kernel name in kprobe must match exactly (including C++ mangling)

**High baseline latency**: Check if GPU is in power-saving mode or shared with other processes

**Periodic spikes**: May indicate context switches, PCIe contention, or driver CPU overhead
