
# CUDA eBPF Early Exit Example

This example demonstrates how to use `bpftime` to exit CUDA kernels even before entering it, which is similar to the role of XDP for network packets, allowing you to:

- Implement the kernel atomizer proposed in [LithOS](https://dl.acm.org/doi/10.1145/3731569.3764818)
- Configure the CUDA thread block partition from the host side.

## Overview

The example consists of two main components:

1. **Vector Addition CUDA Application** (`vec_add`): A simple CUDA application that repeatedly performs vector addition on the GPU.

2. **eBPF CUDA Atomizer** (`cuda_probe`): An eBPF program that attaches to CUDA kernel functions, deciding whether to exit before launching the real kernels.

## How It Works

This example leverages bpftime's CUDA attachment implementation to:

1. Intercept CUDA binary loading via the CUDA runtime API
2. Insert eBPF code (converted to PTX) at the entry point of the `vectorAdd` kernel
3. Execute the eBPF program whenever the kernel is called
4. Exit the thread blocks if the prediction is fulfilled.

## Building the Example

```bash
# Navigate to the bpftime root directory
cd bpftime

# Build the main bpftime project first
cmake -Bbuild -DBPFTIME_ENABLE_CUDA_ATTACH=1 -DBPFTIME_CUDA_ROOT=/usr/local/cuda .
cmake --build build -j$(nproc)

# Build the example
make -C example/gpu/atomizer
```

## Running the Example

You need to start two processes:

### 1. Launch the eBPF Program (Server)

```bash
BPFTIME_LOG_OUTPUT=console LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  example/gpu/atomizer/atomizer
```

This process loads the eBPF program and waits for CUDA events.

### 2. Run the CUDA Application (Client)

In another terminal:

```bash
BPFTIME_LOG_OUTPUT=console LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  example/gpu/atomizer/vec_add
```

This runs the vector addition program with the bpftime agent, which connects to the first process for eBPF execution.

## Understanding the Output

When running successfully, you'll see output like (from the application itself):

```
Exited _Z9vectorAddPKfS0_Pf block_id=1, L=5, H=10
Enter _Z9vectorAddPKfS0_Pf block_id=7, L=5, H=10
C[1] = 0 (expected 0)
C[7] = 21 (expected 21)
```

This shows:
- The kernel entering/exiting logs.
- Vector addition results, with only the higher half computed (10 blocks in total)

## Code Components

### CUDA Vector Addition (`vec_add.cu`)

A simple CUDA application that:
- Allocates memory on GPU and CPU
- Executes a basic vector addition kernel in a loop
- Uses constant memory for vector size

### eBPF Program (`atomizer.bpf.c`) 

Contains an eBPF program:
- `probe__cuda`: Executes when entering the CUDA kernel
  - Load the configurations from the bpf map.
  - Check the prediction for early exit
  
### Userspace Loader (`atomizer.c`)

Manages the eBPF program lifecycle:
- Loads the compiled eBPF code
- Attaches to the CUDA kernel functions
- Handles proper signal termination

## Advanced Features

This example demonstrates several advanced bpftime capabilities:

1. **Custom CUDA Helpers**: Special eBPF helper functions for CUDA:
   - `bpf_cuda_exit()` - Exit the thread block
   - `bpf_get_block_idx()` - Get current block indices
   - `bpf_get_grid_dim()` - Get current grid sizes

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
