# WebAssembly URL Filter Plugin

This directory contains a WebAssembly-based implementation of the URL filter that can be loaded by the `dynamic_load_plugin` in NGINX. The implementation consists of two main components:

1. **WebAssembly Module** (`url_filter.c`): This is compiled to WebAssembly and contains the actual URL filtering logic.
2. **Runtime Wrapper** (`wasm_runtime.c`): This is a native C library that loads and executes the WebAssembly module, implementing the same API as the other filter libraries.

## Prerequisites

To build and run the WebAssembly filter, you need:

- C/C++ compiler (GCC or Clang)
- CMake (for building WAMR)
- Git (for cloning repositories)
- WASI-SDK (for compiling to WebAssembly)

## Building the WebAssembly Filter

Follow these steps to build the WebAssembly filter:

### 1. Install the WASI-SDK

If you don't have WASI-SDK installed, you can use the provided target:

```bash
# Install WASI-SDK to /opt/wasi-sdk
make install-deps
```

Or manually install it:

```bash
wget https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-19/wasi-sdk-19.0-linux.tar.gz
tar xf wasi-sdk-19.0-linux.tar.gz
sudo mv wasi-sdk-19.0 /opt/wasi-sdk
```

### 2. Build the WebAssembly Module and Runtime Wrapper

```bash
cd /path/to/bpftime/example/attach_implementation/benchmark/wasm_plugin

# Build the WebAssembly module and runtime wrapper
make
```

This will:
- Clone and build the WebAssembly Micro Runtime (WAMR)
- Compile the `url_filter.c` to WebAssembly (producing `url_filter.wasm`)
- Compile the `wasm_runtime.c` to a shared library (producing `libwasm_filter.so`)

## Running the WebAssembly Filter with NGINX

To run NGINX with the WebAssembly filter:

```bash
cd /path/to/bpftime/example/attach_implementation

# Start NGINX with the dynamic load module
export DYNAMIC_LOAD_LIB_PATH="$(pwd)/benchmark/wasm_plugin/libwasm_filter.so"
export DYNAMIC_LOAD_URL_PREFIX="/aaaa"
export WASM_MODULE_PATH="$(pwd)/benchmark/wasm_plugin/url_filter.wasm"
./nginx_plugin_output/nginx -p $(pwd) -c benchmark/dynamic_load_module.conf
```

## Testing the WebAssembly Filter

Once NGINX is running with the WebAssembly filter, you can test it using curl:

```bash
# This should succeed (HTTP 200)
curl http://localhost:9026/aaaa

# This should fail (HTTP 403 Forbidden)
curl http://localhost:9026/forbidden_path
```

## How It Works

1. The WebAssembly module (`url_filter.wasm`) contains the following exported functions:
   - `initialize`: Sets the URL prefix to accept and resets counters
   - `url_filter`: Checks if a URL starts with the accepted prefix
   - `get_counters`: Returns the number of accepted and rejected requests
   - `set_buffer`: Sets data in a buffer
   - `get_buffer`: Gets data from a buffer

2. The runtime wrapper (`libwasm_filter.so`):
   - Uses WebAssembly Micro Runtime (WAMR) to execute the WebAssembly module
   - Loads the WebAssembly module from a file specified by the `WASM_MODULE_PATH` environment variable
   - Provides the same API as other filter implementations
   - Translates calls between the native API and the WebAssembly module

3. The `dynamic_load_plugin` in NGINX loads the runtime wrapper as a shared library and calls its functions to:
   - Initialize the filter with a URL prefix
   - Filter requests based on their URLs
   - Keep track of accepted and rejected requests

## Including in Benchmarks

The WebAssembly filter is included in the benchmark script. To run the benchmark:

```bash
# Build the WebAssembly filter first
cd /path/to/bpftime/example/attach_implementation/benchmark/wasm_plugin
make

# Run the benchmark
cd /path/to/bpftime
python3 example/attach_implementation/benchmark/run_benchmark.py
```

The benchmark script will automatically build and use the WebAssembly filter when testing.

## Troubleshooting

### WASI-SDK Not Found

If you get an error about WASI-SDK not being found:

```
WASI-SDK not found at /opt/wasi-sdk
```

Make sure to install WASI-SDK first or set the environment variable to its location:

```bash
export WASI_SDK_PATH=/path/to/your/wasi-sdk
```

### WebAssembly Module Not Found

If you get errors about the WebAssembly module not being found:

```
Failed to open WebAssembly file: url_filter.wasm
```

Make sure to set the `WASM_MODULE_PATH` environment variable to the absolute path of the WebAssembly module.

```bash
export WASM_MODULE_PATH="/absolute/path/to/url_filter.wasm"
```

### Shared Library Not Found

If NGINX can't load the shared library:

```
Failed to load shared library: cannot open shared object file: No such file or directory
```

Make sure the `DYNAMIC_LOAD_LIB_PATH` environment variable is set to the absolute path of the shared library.

```bash
export DYNAMIC_LOAD_LIB_PATH="/absolute/path/to/libwasm_filter.so"
``` 