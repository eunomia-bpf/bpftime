# CUDA Vector Addition Benchmark with BPF and NVBit Probes

This benchmark demonstrates a simple CUDA vector addition operation with eBPF and NVBit probes attached to monitor its execution.

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
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  benchmark/cuda/cuda_probe
# in another terminal
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  benchmark/cuda/vec_add
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

## Basic results

| baseline | bpf | nvbit |
|----------|-----|-------|
| 51.8076 us | 81.0824 us | 174.412 us |

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
#        CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=./nvbit_vec_add.so ./vec_add 
------------- NVBit (NVidia Binary Instrumentation Tool v1.7.5) Loaded --------------
NVBit core environment variables (mostly for nvbit-devs):
ACK_CTX_INIT_LIMITATION = 0 - if set, no warning will be printed for nvbit_at_ctx_init()
            NVDISASM = nvdisasm - override default nvdisasm found in PATH
            NOBANNER = 0 - if set, does not print this banner
       NO_EAGER_LOAD = 0 - eager module loading is turned on by NVBit to prevent potential NVBit tool deadlock, turn it off if you want to use the lazy module loading feature
---------------------------------------------------------------------------------
NVBit: Minimal Vector Addition Instrumentation Tool
------------------------------------------------
WARNING: Do not call CUDA memory allocation in nvbit_at_ctx_init(). It will cause deadlocks. Do them in nvbit_tool_init(). If you encounter deadlocks, remove CUDA API calls to debug.
Running benchmark with 10000 iterations...
Benchmark results:
Total time: 1.74412e+06 us
Average kernel time: 174.412 us
Validation check: C[0] = 0, C[1] = 3

NVBit Instrumentation Summary:
Total kernel calls: 10001
Total execution time: 0.000 ms
Average kernel time: 0.000 us
```

BPF:

```console
# LD_PRELOAD=build/runtime/agent/libbpftime-agent.so   benchmark/cuda/vec_add
[2025-06-06 07:47:25.901] [info] [bpftime_shm_internal.cpp:874] Registered shared memory with CUDA: addr=0x79d3d7200000 size=20971520
[2025-06-06 07:47:25.901] [info] [bpftime_shm_internal.cpp:714] Global shm constructed. shm_open_type 1 for bpftime_maps_shm
[2025-06-06 07:47:25.901] [info] [bpftime_shm_internal.cpp:40] Global shm initialized
[2025-06-06 07:47:25][info][3396212] Initializing CUDA shared memory
[2025-06-06 07:47:28][info][3396212] CUDA context created
[2025-06-06 07:47:28][info][3396212] bpf_attach_ctx constructed
[2025-06-06 07:47:28][info][3396261] CUDA watcher thread started
[2025-06-06 07:47:28][info][3396212] Register attach-impl defined helper bpf_get_func_arg, index 183
[2025-06-06 07:47:28][info][3396212] Register attach-impl defined helper bpf_get_func_ret_id, index 184
[2025-06-06 07:47:28][info][3396212] Register attach-impl defined helper bpf_get_retval, index 186
[2025-06-06 07:47:28][info][3396212] Starting nv_attach_impl
[2025-06-06 07:47:28][info][3396212] Initializing agent..
[2025-06-06 07:47:28][info][3396212] Skipping nv attach handler 8 since we are not handling nv handles
[2025-06-06 07:47:28][info][3396212] Skipping nv attach handler 11 since we are not handling nv handles
[2025-06-06 07:47:28][info][3396212] Main initializing for handlers done, try to initialize cuda link handles....
[2025-06-06 07:47:28][info][3396212] Handling link to CUDA program: 8, recording it..
[2025-06-06 07:47:28][info][3396212] Loaded 2 instructions (original) for cuda ebpf program
[2025-06-06 07:47:28][info][3396212] Recording kprobe for _Z9vectorAddPKfS0_Pf
[2025-06-06 07:47:28][info][3396212] Handling link to CUDA program: 11, recording it..
[2025-06-06 07:47:28][info][3396212] Loaded 2 instructions (original) for cuda ebpf program
[2025-06-06 07:47:28][info][3396212] Recording kretprobe for _Z9vectorAddPKfS0_Pf
[2025-06-06 07:47:28][info][3396212] Attach successfully
[2025-06-06 07:47:28][info][3396212] Got CUBIN section header size = 16, size = 3200
[2025-06-06 07:47:28][info][3396212] Finally size = 3216
[2025-06-06 07:47:28][info][3396212] Temporary fatbin written to /tmp/bpftime-fatbin-store.fN4PE3/temp.fatbin
[2025-06-06 07:47:28][info][3396212] Listing functions in the patched ptx
[2025-06-06 07:47:29][info][3396212] Extracted PTX at /tmp/bpftime-fatbin-store.fN4PE3/temp.ptx
[2025-06-06 07:47:29][info][3396212] Patching with kprobe/kretprobe
[2025-06-06 07:47:29][info][3396212] Compiling eBPF to PTX __probe_func___Z9vectorAddPKfS0_Pf, eBPF instructions count 2, with arguments false
[2025-06-06 07:47:29][info][3396212] Patching with kprobe/kretprobe
[2025-06-06 07:47:29][info][3396212] Compiling eBPF to PTX __retprobe_func___Z9vectorAddPKfS0_Pf, eBPF instructions count 2, with arguments false
[2025-06-06 07:47:29][info][3396212] Recompiling PTX with nvcc..
[2025-06-06 07:47:29][info][3396212] Working directory: /tmp/bpftime-recompile-nvcc
[2025-06-06 07:47:29][info][3396212] PTX IN: /tmp/bpftime-recompile-nvcc/main.ptx
[2025-06-06 07:47:29][info][3396212] Fatbin out /tmp/bpftime-recompile-nvcc/out.fatbin
[2025-06-06 07:47:29][info][3396212] Starting nvcc: nvcc -O2 -G -g --keep-device-functions -arch=sm_60 /tmp/bpftime-recompile-nvcc/main.ptx -fatbin -o /tmp/bpftime-recompile-nvcc/out.fatbin
ptxas warning :  .debug_abbrev section not found
ptxas warning :  .debug_info section not found
[2025-06-06 07:47:29][info][3396212] NVCC execution done.
[2025-06-06 07:47:29][info][3396212] Got patched fatbin in 25064 bytes
[2025-06-06 07:47:29][info][3396212] Registering trampoline memory
[2025-06-06 07:47:29][info][3396212] Register trampoline memory done
[2025-06-06 07:47:29][info][3396212] Copying data to device symbols..
[2025-06-06 07:47:29][info][3396212] Copying the followed map info:
[2025-06-06 07:47:29][info][3396212] constData and map_basic_info copied..
Running benchmark with 10000 iterations...
Benchmark results:
Total time: 810824 us
Average kernel time: 81.0824 us
Validation check: C[0] = 0, C[1] = 3
```