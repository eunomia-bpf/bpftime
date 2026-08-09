# RLBox Sandbox URL Filter Plugin

This directory contains an implementation of the URL filter using the RLBox sandboxing framework. RLBox provides a way to safely sandbox third-party libraries with minimal performance overhead.

## Implementation Variants

This implementation provides two sandboxing options:

1. **NoOp Sandbox**: A lightweight sandbox used during development and initial testing. It doesn't actually enforce isolation but provides the same API as production sandboxes, making it easier to port code to RLBox.

2. **Wasm2c Sandbox**: A production-grade sandbox that compiles the filter library to WebAssembly and then to C, providing strong isolation with good performance.

## Prerequisites

To build and run the RLBox sandboxed filter, you need:

- C/C++ compiler with C++17 support (GCC or Clang)
- RLBox framework
- For wasm2c sandbox: WASI-SDK for WebAssembly compilation
- CMake (for building dependencies)

## Building the Plugin

### 1. Install Dependencies

First, install RLBox and other dependencies:

```bash
cd /path/to/bpftime/example/attach_implementation/benchmark/rlbox_plugin

# Install RLBox and its dependencies
make install-deps
```

### 2. Build the NoOp Sandbox Version (Default)

For development and testing, build the NoOp sandbox version (which is the default):

```bash
cd /path/to/bpftime/example/attach_implementation/benchmark/rlbox_plugin

# Build the NoOp sandbox version
make
```

This will create `libfilter_rlbox.so` which can be used with the dynamic load module.

### 3. Build the Wasm2c Sandbox Version (Production)

For production use with real isolation, build the wasm2c sandbox version:

```bash
cd /path/to/bpftime/example/attach_implementation/benchmark/rlbox_plugin

# Build the wasm2c sandbox version
make wasm2c
```

This will create `libfilter_rlbox_wasm2c.so` which provides strong isolation via WebAssembly.

## Running with Nginx

To run Nginx with the RLBox sandboxed filter:

### Using the RLbox Sandbox

```bash
cd /path/to/bpftime/example/attach_implementation

# Set the environment variables
DYNAMIC_LOAD_LIB_PATH="$(pwd)/benchmark/rlbox_plugin/libfilter_rlbox.so"  DYNAMIC_LOAD_URL_PREFIX="/aaaa" ./nginx_plugin_output/nginx -p $(pwd) -c benchmark/dynamic_load_module.conf
```

### Using the Wasm2c Sandbox

```bash
cd /path/to/bpftime/example/attach_implementation

# Set the environment variables
export DYNAMIC_LOAD_LIB_PATH="$(pwd)/benchmark/rlbox_plugin/libfilter_rlbox_wasm2c.so"
export DYNAMIC_LOAD_URL_PREFIX="/aaaa"

# Start Nginx with the dynamic load module
./nginx_plugin_output/nginx -p $(pwd) -c benchmark/dynamic_load_module.conf
```

## Testing the Filter

Once Nginx is running with the RLBox filter, you can test it using curl:

```bash
# This should succeed (HTTP 200)
curl http://localhost:9026/aaaa

# This should fail (HTTP 403 Forbidden)
curl http://localhost:9026/forbidden_path
```

## How It Works

1. The RLBox framework provides a way to safely interact with untrusted code by:
   - Placing library code in a sandbox
   - Ensuring all data crossing the sandbox boundary is properly validated
   - Preventing memory safety violations from escaping the sandbox

2. Our implementation:
   - Consists of a filter library (`mylib.c`) that contains the filtering logic
   - Wraps this library with RLBox sandbox (`rlbox_filter.cpp`)
   - Provides the same module_* API as other filter implementations

3. The NoOp Sandbox:
   - Doesn't actually enforce isolation
   - Used for development and testing
   - Helps with porting code to use RLBox

4. The Wasm2c Sandbox:
   - Compiles the filter library to WebAssembly
   - Converts WebAssembly to C using wasm2c
   - Provides strong isolation with good performance

## Including in Benchmarks

The RLBox filter can be included in the benchmark script like other implementations. To run the benchmark:

```bash
# Build the RLBox filter first
cd /path/to/bpftime/example/attach_implementation/benchmark/rlbox_plugin
make

# Run the benchmark
cd /path/to/bpftime
python3 example/attach_implementation/benchmark/run_benchmark.py
```

## Troubleshooting

### RLBox Not Found

If you get errors about RLBox not being found:

```
fatal error: rlbox/rlbox.hpp: No such file or directory
```

Make sure to run `make install-deps` or manually clone the RLBox repositories:

```bash
git clone https://github.com/PLSysSec/rlbox
git clone https://github.com/PLSysSec/rlbox_wasm2c_sandbox
```

### WASI-SDK Not Found

If you get errors about WASI-SDK not being found when building the wasm2c version:

```
WASI-SDK not found at /opt/wasi-sdk
```

Run `make install-wasi-sdk` or manually install WASI-SDK.

### Shared Library Not Found

If Nginx can't load the shared library:

```
Failed to load shared library: cannot open shared object file: No such file or directory
```

Make sure the `DYNAMIC_LOAD_LIB_PATH` environment variable is set to the absolute path of the shared library.

## References

- [RLBox Documentation](https://rlbox.dev/)
- [RLBox GitHub Repository](https://github.com/PLSysSec/rlbox)
- [RLBox Wasm2c Sandbox](https://github.com/PLSysSec/rlbox_wasm2c_sandbox) 