# NVBit Instrumentation Tool for CUDA Benchmarks

This directory contains an NVBit-based instrumentation tool for monitoring CUDA kernel execution with minimal overhead.

## Overview

NVBit (NVIDIA Binary Instrumentation Tool) is a research prototype that allows dynamic instrumentation of CUDA applications by intercepting CUDA driver API calls and optionally injecting code into kernels at the SASS level.

This tool provides:
- Kernel launch counting
- Per-kernel execution timing using CUDA events
- Summary statistics at application termination

## Files

- **nvbit_vec_add.cu** - Main NVBit instrumentation tool (host code)
- **nvbit_timing_funcs.cu** - Device functions for instrumentation (currently unused)
- **Makefile** - Build configuration for the NVBit tool

## Prerequisites

- **NVBit**: Download from [https://github.com/NVlabs/NVBit](https://github.com/NVlabs/NVBit)
  - Expected location: `~/nvbit_release_x86_64/`
  - If installed elsewhere, set `NVBIT_PATH` in the Makefile
- **CUDA Toolkit**: Version 11.0 or later
- **GCC**: C++11 compatible compiler

## Building

To build the NVBit instrumentation tool:

```bash
cd /path/to/bpftime/benchmark/gpu/nvbit
make
```

This will produce `nvbit_vec_add.so`, a shared library that can be preloaded to instrument CUDA applications.

### Build Options

- **DEBUG=1**: Build with debug symbols and no optimization
- **ARCH=sm_XX**: Target specific GPU architecture (default: `all`)
- **NVBIT_PATH**: Override NVBit installation path

Example:
```bash
make DEBUG=1 ARCH=sm_75
```

## Usage

### Basic Usage

To instrument a CUDA application, use `LD_PRELOAD`:

```bash
export LD_PRELOAD=/path/to/nvbit_vec_add.so
./your_cuda_application
```

Or in one line:

```bash
LD_PRELOAD=/path/to/nvbit/nvbit_vec_add.so ./your_cuda_application
```

### Testing with Workload Benchmarks

Test against the vec_add benchmark:

```bash
cd ../workload
export LD_PRELOAD=/home/yunwei37/workspace/bpftime/benchmark/gpu/nvbit/nvbit_vec_add.so
./vec_add
```

Test against the matrixMul benchmark:

```bash
cd ../workload
export LD_PRELOAD=/home/yunwei37/workspace/bpftime/benchmark/gpu/nvbit/nvbit_vec_add.so
./matrixMul
```

### Environment Variables

NVBit supports several environment variables (see NVBit banner output):

- `NOBANNER=1` - Suppress NVBit banner
- `TOOL_VERBOSE=1` - Enable verbose output (if implemented)
- `NVDISASM=/path/to/nvdisasm` - Override nvdisasm location

## Expected Output

When the tool is loaded, you should see:

```
------------- NVBit (NVidia Binary Instrumentation Tool v1.7.6) Loaded --------------
NVBit: Minimal Vector Addition Instrumentation Tool
------------------------------------------------
```

For each kernel launch (if instrumentation is working):

```
NVBit: Kernel kernel_name - Time: XXX.XXX us
```

At program termination:

```
NVBit Instrumentation Summary:
Total kernel calls: N
Total execution time: XXX.XXX ms
Average kernel time: XXX.XXX us
```

## Current Limitations

**Note**: The current implementation successfully loads and allows CUDA applications to run without crashing, but kernel launch callbacks are not being triggered in the Runtime API compilation mode used by the workload benchmarks. This is a known limitation of the current implementation.

### Why Kernel Counting Doesn't Work

The workload benchmarks (`vec_add`, `matrixMul`) are compiled with `-cudart shared`, which uses the CUDA Runtime API. The current NVBit tool only intercepts CUDA Driver API calls (`cuLaunchKernel`). The Runtime API wraps the Driver API, and in newer CUDA versions with shared runtime, the callbacks may not be triggered as expected.

### Possible Solutions

1. **Use Driver API benchmarks**: Recompile benchmarks to use the CUDA Driver API directly
2. **Intercept Runtime API**: Modify the tool to intercept `cudaLaunch*` calls
3. **Use NVBit device instrumentation**: Inject actual device code to count kernel launches (requires proper device code injection)

## Performance Overhead

Based on the original benchmark results (see `../README.md`):

| Device | Baseline | NVBit Overhead |
|--------|----------|----------------|
| **NVIDIA P40** | 51.8 μs | 174.4 μs (3.37x) |
| **NVIDIA RTX 5090** | 4.1 μs | 55.8 μs (13.6x) |

The overhead comes from:
- Binary instrumentation of kernels
- Runtime analysis and callback overhead
- Event-based timing measurements

## Comparison with BPFtime

For comparison with eBPF-based instrumentation using bpftime:

| Approach | P40 Overhead | RTX 5090 Overhead | Notes |
|----------|--------------|-------------------|-------|
| **NVBit** | 3.37x | 13.6x | Binary instrumentation, GPU-level |
| **BPFtime** | 1.56x | 2.0x | System-level hooks, lower overhead |

## Cleaning Up

To remove build artifacts:

```bash
make clean
```

## Troubleshooting

### Tool loads but shows 0 kernel calls

This is the current known issue. The tool loads successfully but doesn't intercept Runtime API kernel launches. See "Current Limitations" above.

### Segmentation fault

- Ensure NVBit version matches your CUDA version
- Check that CUDA toolkit is properly installed
- Verify GPU driver version compatibility

### Build errors

- Verify `NVBIT_PATH` points to correct NVBit installation
- Ensure CUDA toolkit is in PATH
- Check GCC version is compatible with your CUDA version

## References

- [NVBit GitHub Repository](https://github.com/NVlabs/NVBit)
- [NVBit Paper](https://ieeexplore.ieee.org/document/8891668)
- [CUDA Driver API Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/)

## License

This tool follows the bpftime project license. NVBit itself is provided under the BSD-3-Clause license by NVIDIA.
