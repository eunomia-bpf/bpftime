# Multi-GPU CUDA attach example

This example runs the same CUDA kernel on multiple GPUs and counts its launches
per device from an eBPF probe.

## Build

```bash
make
```

## Run

```bash
# Use every available GPU, or pass a device count.
./multi_gpu_vec_add
./multi_gpu_vec_add 2

# Load the probe, then run the workload through bpftime.
bpftime load ./multi_gpu_probe
bpftime start ./multi_gpu_vec_add
```

The probe uses helper 512 to read the device ordinal from each patched module.
This exercises device enumeration, per-device PTX compilation and module
loading, and device-aware kernel interception.
