
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
- `BPFTIME_ENABLE_GDRCOPY`: Enable optional GDRCopy support for faster host-side GPU map reads (default: OFF). Requires GDRCopy user library (`libgdrapi.so`) and the `gdrdrv` kernel module.

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

## Configuration Reference

This section collects **all** GPU-related macro definitions, CMake options, compile-time macros, and runtime environment variables in one place.

### CMake Build Options

These options are passed to CMake at configure time (e.g., `-DBPFTIME_ENABLE_CUDA_ATTACH=ON`).

| Option | Default | Description |
|--------|---------|-------------|
| `BPFTIME_ENABLE_CUDA_ATTACH` | `OFF` | Enable the CUDA attachment pipeline. Required to build and use any GPU eBPF features. |
| `BPFTIME_CUDA_ROOT` | _(required if `CUDA_PATH` unset)_ | Absolute path to the CUDA installation root (e.g., `/usr/local/cuda-12.6`). **Required** when `BPFTIME_ENABLE_CUDA_ATTACH=ON` if the `CUDA_PATH` environment variable is not set. |
| `BPFTIME_ENABLE_GDRCOPY` | `OFF` | Enable optional [GDRCopy](https://github.com/NVIDIA/gdrcopy) support for faster host-side reads from GPU maps. Requires the GDRCopy user library (`libgdrapi.so`) and the `gdrdrv` kernel module. Falls back to `cuMemcpyDtoH` automatically if unavailable at runtime. |

**Example – minimal CUDA build:**

```bash
cmake -S . -B build \
  -DBPFTIME_ENABLE_CUDA_ATTACH=ON \
  -DBPFTIME_CUDA_ROOT=/usr/local/cuda-12.6
cmake --build build -j$(nproc)
```

**Example – with GDRCopy:**

```bash
cmake -S . -B build \
  -DBPFTIME_ENABLE_CUDA_ATTACH=ON \
  -DBPFTIME_CUDA_ROOT=/usr/local/cuda-12.6 \
  -DBPFTIME_ENABLE_GDRCOPY=ON
cmake --build build -j$(nproc)
```

---

### Compile-time Preprocessor Macros

These macros are automatically set by CMake when the corresponding option is enabled.  They guard GPU-specific code paths throughout the runtime.

| Macro | Set by | Description |
|-------|--------|-------------|
| `BPFTIME_ENABLE_CUDA_ATTACH` | `-DBPFTIME_ENABLE_CUDA_ATTACH=ON` | Guards all CUDA-specific code in the runtime, agent, syscall-server, daemon, and tools. |
| `BPFTIME_ENABLE_GDRCOPY` | `-DBPFTIME_ENABLE_GDRCOPY=ON` | Guards GDRCopy acceleration code paths in `runtime/src/bpf_map/gpu/nv_gpu_gdrcopy.cpp`. |
| `BPFTIME_HAVE_GDRAPI_H` | Auto-detected (presence of `<gdrapi.h>`) | Set automatically when the GDRCopy header is found during compilation. Enables native `gdr_t`/`gdr_mh_t` type usage instead of the bundled stub declarations. |
| `BPFTIME_ENABLE_ROCM_ATTACH` | _(future/in-development)_ | Guards ROCm/HIP attachment code paths. Not yet wired to a CMake option; present for forward compatibility. |

---

### Runtime Environment Variables

These variables are read at process startup (no rebuild required).

#### GPU Map & Thread Count

| Variable | Default | Description |
|----------|---------|-------------|
| `BPFTIME_MAP_GPU_THREAD_COUNT` | _(auto from map attributes)_ | Override the maximum GPU thread count for all GPU maps (e.g., `BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP`, ringbuf maps). Useful when the auto-computed value is too large and causes shared-memory allocation failures. Example: `BPFTIME_MAP_GPU_THREAD_COUNT=8192`. |

#### CUDA Compilation & Architecture

| Variable | Default | Description |
|----------|---------|-------------|
| `BPFTIME_SM_ARCH` | _(auto-detected via `cuCtxGetDevice`)_ | Override the CUDA SM (streaming-multiprocessor) architecture string used when recompiling patched PTX with `nvcc` (e.g., `sm_80`, `sm_90`). Useful when auto-detection fails or when cross-compiling for a different GPU. |
| `BPFTIME_CUDA_ROOT` | _(from CMake `-DBPFTIME_CUDA_ROOT`)_ | When set as a shell environment variable, points `nvcc`/`cuobjdump` invocations at a specific CUDA installation directory at runtime. This is used by build scripts and the PTX recompilation path; it does **not** override the value baked in by CMake – set it if invoking bpftime tools outside of the normal CMake-built binary (e.g., in wrapper scripts). |

#### PTX Pass Pipeline

| Variable | Default | Description |
|----------|---------|-------------|
| `BPFTIME_PTXPASS_LIBRARIES` | _(bundled default paths)_ | Colon-separated list of PTX-pass shared library (`.so`) paths. When set, overrides the default bundled pass executables. Example: `BPFTIME_PTXPASS_LIBRARIES=/path/to/libptxpass_kprobe.so:/path/to/libptxpass_kretprobe.so`. |

#### Internal Variables (for reference only)

The following variables are set **internally** by the bpftime runtime and are documented here for completeness. They are not intended to be set by end users.

| Variable | Description |
|----------|-------------|
| `PTX_ATTACH_POINT` | Set by the runtime when invoking an external PTX pass process to indicate the current attachment point being processed (`kprobe`, `kretprobe`, `memcapture`). |

#### GDRCopy Tuning (host-side GPU map reads)

| Variable | Default | Description |
|----------|---------|-------------|
| `BPFTIME_GPU_ARRAY_GDRCOPY` | `0` | Set to `1` to enable GDRCopy for host-side lookup of GPU array maps. Has no effect if `BPFTIME_ENABLE_GDRCOPY` was not set at build time, or if GDRCopy is unavailable at runtime (falls back to `cuMemcpyDtoH` silently). |
| `BPFTIME_GPU_ARRAY_GDRCOPY_MAX_PER_KEY_BYTES` | `4096` | Skip GDRCopy and fall back to `cuMemcpyDtoH` when the per-key copy size exceeds this threshold (bytes). Set to `0` to disable the threshold (always use GDRCopy when enabled). |

---

### Quick Reference

```bash
# Build with all GPU features
cmake -S . -B build \
  -DBPFTIME_ENABLE_CUDA_ATTACH=ON \
  -DBPFTIME_CUDA_ROOT=/usr/local/cuda-12.6 \
  -DBPFTIME_ENABLE_GDRCOPY=ON

# Run with overrides at runtime
BPFTIME_MAP_GPU_THREAD_COUNT=8192 \
BPFTIME_SM_ARCH=sm_80 \
BPFTIME_GPU_ARRAY_GDRCOPY=1 \
BPFTIME_GPU_ARRAY_GDRCOPY_MAX_PER_KEY_BYTES=8192 \
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  ./my_cuda_probe
```
