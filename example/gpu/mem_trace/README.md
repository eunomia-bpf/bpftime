# mem_trace

A simple example to trace CUDA kernel invocations using eBPF.

This example demonstrates how to use bpftime to attach eBPF probes to CUDA kernels running in userspace and trace their execution.

## Build

```bash
# Build bpftime first (from the bpftime root directory)
cd bpftime
make release

# Build the example
make -C example/gpu/mem_trace
```

This will build:
- `mem_trace` - The eBPF tracing program
- `vec_add` - CUDA vector addition victim program (requires nvcc)

## Run

You need to start two processes:

### 1. Launch the eBPF Program (Server)

```bash
BPFTIME_LOG_OUTPUT=console LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  example/gpu/mem_trace/mem_trace
```

This process loads the eBPF program and waits for CUDA events.

### 2. Run the CUDA Application (Client)

In another terminal:

```bash
BPFTIME_LOG_OUTPUT=console  LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  example/gpu/mem_trace/vec_add
```

This runs the vector addition program with the bpftime agent, which connects to the first process for eBPF execution.

## Understanding the Output

The mem_trace program will print statistics showing the number of CUDA kernel invocations per process:

```
16:30:45
	pid=12345 	mem_traces: 120
```

This shows that process 12345 has executed the CUDA kernel 120 times.
