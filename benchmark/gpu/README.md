# CUDA Vector Addition Benchmark with BPF and NVBit Probes

This benchmark demonstrates a simple CUDA vector addition operation with eBPF and NVBit probes attached to monitor its execution.

## Performance Results (10,000 iterations)

| Device | Baseline | BPF (bpftime) | NVBit |
|--------|----------|---------------|-------|
| **NVIDIA P40** | 51.8 μs | 81.1 μs (1.56x) | 174.4 μs (3.37x) |
| **NVIDIA RTX 5090** | 4.1 μs | 8.2 μs (2.0x) | 55.8 μs (13.6x) |

## Overview

The benchmark consists of the following components:

1. **vec_add.cu**: A CUDA kernel that performs vector addition with configurable iterations.
2. **cuda_probe.bpf.c**: An eBPF program that attaches to the CUDA kernel and monitors:
   - Number of kernel invocations
   - Total execution time
   - Average execution time per invocation
3. **cuda_probe.c**: A userspace program that loads and attaches the eBPF program.
4. **nvbit_vec_add.cu** and **nvbit_timing_funcs.cu**: An NVBit instrumentation tool that provides similar monitoring using the NVBit framework.

## Prerequisites

- CUDA Toolkit (tested with CUDA 11.x and above)
- LLVM/Clang with BPF target support
- libbpf
- NVBit (for the NVBit instrumentation option)

## Building

To build all components:

```bash
make
```

This will build:
- The `vec_add` CUDA benchmark
- The `cuda_probe.bpf.o` BPF object file
- The `cuda_probe` userspace loader
- The `nvbit_vec_add.so` NVBit instrumentation tool

## Running the Benchmark

### Running with BPF Probes

1. To run the benchmark with BPF probes attached:

```bash
BPFTIME_LOG_OUTPUT=console LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  benchmark/gpu/micro/cuda_probe
# in another terminal
BPFTIME_LOG_OUTPUT=console LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  benchmark/gpu/micro/vec_add
```

This will:
- Start the BPF probe in the background
- Run the CUDA benchmark with 100 iterations
- Show the probe statistics
- Terminate the probe process

2. To run each component separately:

First, start the BPF probe:
```bash
./cuda_probe
```

In another terminal, run the benchmark:
```bash
./vec_add [iterations]
```

Where `iterations` is an optional parameter (default: 10) specifying the number of kernel launches.

### Running with NVBit Instrumentation

1. To run the benchmark with NVBit instrumentation:

```bash
make run_nvbit
```

This will:
- Load the NVBit instrumentation tool
- Run the CUDA benchmark with 100 iterations
- Print timing information for each kernel invocation
- Show a summary of kernel statistics at the end

2. For more verbose output:

```bash
make run_nvbit_verbose
```

This will show additional information during execution.

## Expected Output

### CUDA Benchmark Output

The CUDA benchmark will output timing information:
```
Running benchmark with N iterations...
Benchmark results:
Total time: XXX ms
Average kernel time: XXX ms
Validation check: C[0] = 0, C[1] = 3
```

### BPF Probe Output

The BPF probe will continuously output:
```
PID XXX: kernel calls = N, total time = XXX ns, avg = XXX ns
```

### NVBit Instrumentation Output

The NVBit tool will show per-kernel timing:
```
NVBit: Minimal Vector Addition Instrumentation Tool
------------------------------------------------
NVBit: Kernel _Z9vectorAddPKfS0_Pf - Time: XXX.XXX us
...

NVBit Instrumentation Summary:
Total kernel calls: XXX
Total execution time: XXX.XXX ms
Average kernel time: XXX.XXX us
```

## Comparison

The benchmark allows comparison between BPF and NVBit approaches for instrumenting CUDA applications:

1. **BPF**: Uses Linux's eBPF infrastructure to attach to kernel functions, providing system-level monitoring with minimal overhead.

2. **NVBit**: Uses NVIDIA's binary instrumentation tool to modify the SASS code of CUDA kernels at runtime, offering more detailed GPU-specific information but with potentially higher overhead.

## Cleaning Up

To clean all build artifacts:

```bash
make clean
``` 

##only results

Device: NVIDIA P40

baseline:

```console
#   benchmark/cuda/vec_add
Running benchmark with 10000 iterations...
Benchmark results:
Total time: 518076 us
Average kernel time: 51.8076 us
Validation check: C[0] = 0, C[1] = 3
```

NVbit:

```console
Running benchmark with 10000 iterations...
Benchmark results:
Total time: 1.74412e+06 us
Average kernel time: 174.412 us
Validation check: C[0] = 0, C[1] = 3
```

BPF:

```console
Running benchmark with 10000 iterations...
Benchmark results:
Total time: 810824 us
Average kernel time: 81.0824 us
Validation check: C[0] = 0, C[1] = 3
```

Device: NVIDIA RTX 5090

No trace:

```
$ benchmark/cuda/vec_add
Running benchmark with 10000 iterations...
Benchmark results:
Total time: 40981.7 us
Average kernel time: 4.09817 us
Validation check: C[0] = 0, C[1] = 3
```

With bpftime:

```console  
Running benchmark with 10000 iterations...
Benchmark results:
Total time: 81883.6 us
Average kernel time: 8.18836 us
Validation check: C[0] = 0, C[1] = 3
```


with NVbit:

```console
$ make run_nvbit_verbose
CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=./nvbit_vec_add.so TOOL_VERBOSE=1 ./vec_add
------------- NVBit (NVidia Binary Instrumentation Tool v1.7.6) Loaded --------------
Running benchmark with 10000 iterations...
Benchmark results:
Total time: 557560 us
Average kernel time: 55.756 us
Validation check: C[0] = 0, C[1] = 3
```