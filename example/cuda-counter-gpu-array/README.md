
# CUDA eBPF Probe/Retprobe Example

This example demonstrates how to use `bpftime` to instrument CUDA kernels with eBPF probes, allowing you to:

- Capture entry and exit points of CUDA kernels
- Measure kernel execution time
- Access CUDA context information (block indices, thread indices)
- Modify kernel behavior (in advanced cases)

## Overview

The example consists of two main components:

1. **Vector Addition CUDA Application** (`vec_add`): A simple CUDA application that repeatedly performs vector addition on the GPU.

2. **eBPF CUDA Probe** (`cuda_probe`): An eBPF program that attaches to CUDA kernel functions, monitoring their execution and timing.

## How It Works

This example leverages bpftime's CUDA attachment implementation to:

1. Intercept CUDA binary loading via the CUDA runtime API
2. Insert eBPF code (converted to PTX) at both the entry and exit points of the `vectorAdd` kernel
3. Execute the eBPF program whenever the kernel is called
4. Provide measurements and insights into CUDA kernel execution

## Building the Example

```bash
# Navigate to the bpftime root directory
cd bpftime

# Build the main bpftime project first
mkdir -p build && cd build
cmake .. -DBPFTIME_ENABLE_CUDA_ATTACH=1 -DBPFTIME_CUDA_ROOT=/usr/local/cuda-12.6
make -j$(nproc)

# Build the example (from the build directory)
cd ..
make -C example/cuda-counter-gpu-array
```

## Running the Example

You need to start two processes:

### 1. Launch the eBPF Program (Server)

```bash
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  example/cuda-counter-gpu-array/cuda_probe
```

This process loads the eBPF program and waits for CUDA events.

### 2. Run the CUDA Application (Client)

In another terminal:

```bash
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  example/cuda-counter-gpu-array/vec_add
```

This runs the vector addition program with the bpftime agent, which connects to the first process for eBPF execution.

## Understanding the Output

When running successfully, you'll see output like:

```
Entered _Z9vectorAddPKfS0_Pf x=0, ts=1749147474550023136
Exited (with tsp) _Z9vectorAddPKfS0_Pf x=0 duration=6437888 tsp=1749147474550023136ns
C[0] = 0 (expected 0)
C[1] = 0 (expected 3)
```

This shows:
- Detection of kernel entry with timestamp
- Detection of kernel exit with execution duration
- Vector addition results (from the application itself)

## Code Components

### CUDA Vector Addition (`vec_add.cu`)

A simple CUDA application that:
- Allocates memory on GPU and CPU
- Executes a basic vector addition kernel in a loop
- Uses constant memory for vector size

### eBPF Program (`cuda_probe.bpf.c`) 

Contains two eBPF programs:
- `probe__cuda`: Executes when entering the CUDA kernel
  - Records timestamp
  - Captures block index information
  - Increments call counter

- `retprobe__cuda`: Executes when exiting the CUDA kernel
  - Calculates execution duration
  - Accumulates total execution time
  - Outputs trace information

### Userspace Loader (`cuda_probe.c`)

Manages the eBPF program lifecycle:
- Loads the compiled eBPF code
- Attaches to the CUDA kernel functions
- Handles proper signal termination

## Advanced Features

This example demonstrates several advanced bpftime capabilities:

1. **Custom CUDA Helpers**: Special eBPF helper functions for CUDA:
   - `bpf_get_globaltimer()` - Access GPU timer
   - `bpf_get_block_idx()` - Get current block indices
   - `bpf_get_thread_idx()` - Get thread indices

2. **Interprocess Communication**: The eBPF program runs in a separate process from the CUDA application, communicating through shared memory.

3. **Dynamic Binary Modification**: The CUDA binary is intercepted, modified, and recompiled at runtime.

## Troubleshooting

If you encounter issues:

- Ensure CUDA is properly installed and in your path
- Check that both processes are running and can communicate
- Verify the PTX modification succeeded in the logs
- If you see CUDA errors, try simplifying the vector addition kernel

## Further Exploration

Try modifying this example to:
- Track memory access patterns in CUDA kernels
- Measure specific operations within kernels
- Apply eBPF programs to more complex CUDA applications
- Implement performance optimizations based on eBPF insights
