
# CUDA eBPF Attachment Implementation (bpftime-nv-attach)

This module provides CUDA attachment support for bpftime, allowing eBPF programs to be injected into and executed within CUDA kernels. It enables dynamic instrumentation, tracing, and modification of GPU code execution without requiring application recompilation or restarts.

## How It Works

The CUDA attachment implementation works through several mechanisms:

1. **CUDA API Interception**:
   - Uses Frida-gum to hook key CUDA runtime functions including `__cudaRegisterFatBinary`, `__cudaRegisterFunction`, `__cudaRegisterFatBinaryEnd`, and `cudaLaunchKernel`
   - Intercepts the binary data before it's passed to the CUDA driver

2. **PTX Code Transformation**:
   - Extracts PTX code from CUDA fat binaries
   - Patches the PTX by inserting instrumentation code at specific locations
   - Converts eBPF programs to PTX using LLVM-based transformations
   - Recompiles the modified PTX back to a binary format using NVCC

3. **Register Protection**:
   - Implements sophisticated register saving/restoration mechanisms
   - Ensures that the original kernel behavior is preserved
   - Handles various PTX register types and patterns

4. **Memory and Data Sharing**:
   - Sets up shared memory between host and device
   - Creates communication channels for passing data between eBPF and CUDA
   - Handles map information for eBPF maps

5. **Attachment Types**:
   - **Memory Capture**: Intercepts memory operations (load/store)
   - **Function Probes**: Executes at the beginning of CUDA kernel functions
   - **Return Probes**: Executes before kernel functions return

## Building

### Prerequisites

- CUDA Toolkit (tested with 12.6, other versions may also work)
- Frida-gum library
- LLVM for eBPF compilation
- C++20 compatible compiler
- CMake 3.10+
- spdlog

### Build Steps

```bash
# Clone the repository if you haven't already
git clone https://github.com/eunomia-bpf/bpftime.git
cd bpftime

# Create and enter build directory
mkdir -p build && cd build

# Configure with CMake
cmake -B build \
  -DBPFTIME_ENABLE_CUDA_ATTACH=1 \
  -DBPFTIME_CUDA_ROOT=/usr/local/cuda-12.6  # Specify your CUDA installation path

# Build
make -j$(nproc) -C build
```

### CMake Options

- `BPFTIME_ENABLE_CUDA_ATTACH`: Enable CUDA attachment support (default: OFF)
- `BPFTIME_CUDA_ROOT`: Path to CUDA installation (required if CUDA_PATH environment variable isn't set)

## Implementation Details

### PTX Transformation

The module transforms eBPF instructions into PTX code by:

1. Generating an initial PTX representation using LLVM
2. Adding register guards to preserve register state
3. Filtering unnecessary headers and sections
4. Wrapping with trampoline code for proper execution

### Communication Channel

Communication between CPU and GPU happens through:

1. A shared memory pointer (`constData`)
2. Map information structures in constant memory
3. Synchronization mechanisms for safe access

### Memory Capture

Memory operation interception works by:

1. Finding load/store operations in PTX using regex patterns
2. Injecting calls to custom functions before these operations
3. Passing memory addresses and data to the eBPF program

## Debugging

For debugging purposes, the code can:

- Dump PTX code to `/tmp` directories
- Print detailed logs about the patching process
- Show information about register usage and state

## Limitations

- Currently only supports specific CUDA version formats
- Requires a compatible NVCC version for recompilation
- May have performance overhead for complex kernels
- Limited to features supported by both eBPF and PTX

## Further Development

Future improvements planned for this module:

- Support for more CUDA versions and features
- Performance optimizations for reduced overhead
- Additional attachment points and capabilities
- Better integration with the bpftime ecosystem
