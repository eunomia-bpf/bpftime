# ERIM-Protected URL Filter Plugin

This directory contains an implementation of a URL filtering plugin protected with ERIM (Efficient Remote Isolation with Memory Protection Keys), providing hardware-enforced isolation between the host application (e.g., Nginx) and the plugin.

## How ERIM Protection Works

ERIM uses Intel's Memory Protection Keys (MPK) to create isolated memory domains with the following features:

1. **Hardware-Enforced Isolation**: 
   - Uses Intel MPK to protect specific memory regions
   - Prevents unauthorized access to protected memory

2. **Domain Switching**:
   - Controlled transitions between trusted and untrusted domains
   - Only specific, validated code can switch domains

3. **Memory Protection**:
   - Global variables are isolated in protected memory
   - String operations occur in the trusted domain

## Prerequisites

To build and run the ERIM-protected filter, you need:

- C compiler (GCC or Clang)
- CPU with Intel MPK support (Intel x86 processors from Skylake onwards)
- Linux kernel with MPK support (4.9+)
- Make

You can check if your CPU supports MPK with:

```bash
grep -q pku /proc/cpuinfo && echo "MPK supported" || echo "MPK not supported"
```

## Building the ERIM-Protected Filter

Follow these steps to build the ERIM-protected filter:

```bash
cd /path/to/bpftime/example/attach_implementation/benchmark/erim_plugin

# Build the ERIM library and filter implementation
make
```

This will:
1. Build the ERIM library (with position-independent code)
2. Compile the filter implementation with ERIM protection
3. Create a shared library `liberim_filter.so`

## Running with Nginx

To run Nginx with the ERIM-protected filter:

```bash
cd /path/to/bpftime/example/attach_implementation

# Set environment variables for the dynamic loader
DYNAMIC_LOAD_LIB_PATH="$(pwd)/benchmark/erim_plugin/liberim_filter.so"  DYNAMIC_LOAD_URL_PREFIX="/aaaa" ./nginx_plugin_output/nginx -p $(pwd) -c benchmark/dynamic_load_module.conf
```

## Testing the ERIM-Protected Filter

### Basic Functionality Test

Once Nginx is running with the ERIM-protected filter, you can test it using curl:

```bash
# This should succeed (HTTP 200)
curl http://localhost:9026/aaaa

# This should fail (HTTP 403 Forbidden)
curl http://localhost:9026/forbidden_path
```

### Standalone Test

You can also run the included test program to verify the ERIM protection:

```bash
cd /path/to/bpftime/example/attach_implementation/benchmark/erim_plugin
make test
```

The test demonstrates:
- Basic URL filtering functionality
- Access to statistics
- Memory protection via ERIM (attempting to modify protected memory will cause a segmentation fault)

## Implementation Details

The plugin uses ERIM to protect its internal state:

- **Global Variables**: `accept_url_prefix` and `counter` are protected in trusted memory
- **API Functions**: Each exported function switches to the trusted domain, performs its operation, then switches back
- **Memory Access**: Direct memory access to protected variables from outside causes a segmentation fault

## Including in Benchmarks

The ERIM-protected filter is included in the benchmark script. To run the benchmark:

```bash
# Build the ERIM filter first
cd /path/to/bpftime/example/attach_implementation/benchmark/erim_plugin
make

# Run the benchmark
cd /path/to/bpftime
python3 example/attach_implementation/benchmark/run_benchmark.py
```

The benchmark script will automatically build and use the ERIM-protected filter when testing.

## Troubleshooting

### MPK Not Supported

If you get an error about MPK not being supported:

```
WRPKRU instruction not supported on this CPU
```

You need a CPU with Intel MPK support (generally Intel processors from Skylake onwards).

### Permission Denied

If you get a permission error:

```
Failed to set protection key for memory region
```

Make sure you have the necessary permissions to use protection keys. You might need to run as root or adjust the kernel parameters.

### Shared Library Not Found

If Nginx can't load the shared library:

```
Failed to load shared library: cannot open shared object file: No such file or directory
```

Make sure the `DYNAMIC_LOAD_LIB_PATH` environment variable is set to the absolute path of the shared library.

```bash
export DYNAMIC_LOAD_LIB_PATH="/absolute/path/to/liberim_filter.so"
```

## Security Benefits

This implementation provides several security benefits:

1. **Isolation**: Plugin state is isolated from the host application
2. **Integrity**: Protected variables cannot be modified directly
3. **Access Control**: Only explicit API functions can access protected state
4. **Hardware Enforcement**: Protection is enforced by the CPU, not software

## Performance Considerations

ERIM provides efficient isolation with minimal overhead:
- Domain switching is a lightweight operation
- Performance impact is primarily from the WRPKRU instruction
- Much lower overhead compared to process-based isolation
