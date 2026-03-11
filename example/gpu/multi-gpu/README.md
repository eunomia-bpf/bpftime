# Multi-GPU eBPF Tracing Example

This example demonstrates bpftime's multi-GPU support for eBPF-based kernel tracing.
It runs a vector addition kernel concurrently on multiple GPUs while eBPF probes
instrument each GPU's execution independently.

## Files

- `multi_gpu_vec_add.cu` - CUDA program that distributes vector addition across multiple GPUs
- `multi_gpu_probe.bpf.c` - eBPF probe that traces `vectorAdd` kernel entry/exit with timing
- `multi_gpu_probe.c` - Userspace loader that prints per-block call counts and execution times

## Build

```bash
# Build the CUDA workload
nvcc -cudart shared multi_gpu_vec_add.cu -o multi_gpu_vec_add -g

# Build the eBPF probe loader
make
```

## Run

```bash
# Run standalone to verify correctness (uses all GPUs by default)
./multi_gpu_vec_add

# Limit to N GPUs
./multi_gpu_vec_add 2

# Run with bpftime GPU attach
export PATH=$PATH:~/.bpftime/
bpftime load ./multi_gpu_vec_add
bpftime start ./multi_gpu_probe
```

## What This Demonstrates

1. **Device enumeration**: bpftime's `gpu_device_manager` detects all available GPUs at startup
2. **Per-device SM architecture**: PTX is compiled for each GPU's compute capability
3. **Per-device module loading**: Patched CUDA modules are loaded into each GPU's context
4. **Per-device kernel tracking**: Patched kernel functions are tracked per-device for correct launch interception
5. **Concurrent multi-GPU tracing**: The same eBPF probe instruments kernel execution on all GPUs simultaneously
