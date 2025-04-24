# WebAssembly URL Filter Plugin

This directory contains a WebAssembly-based implementation of the URL filter that can be loaded by the `dynamic_load_plugin` in NGINX. The implementation consists of two main components:

1. **WebAssembly Module** (`url_filter.c`): This is compiled to WebAssembly and contains the actual URL filtering logic.
2. **Runtime Wrapper** (`wasm_runtime.c`): This is a native C library that loads and executes the WebAssembly module, implementing the same API as the other filter libraries.

## Prerequisites

To build and run the WebAssembly filter, you need:

- C/C++ compiler (GCC or Clang)
- CMake (for building Wasm3)
- Git (for cloning repositories)
- Emscripten SDK (for compiling WebAssembly)

## Building the WebAssembly Filter

Follow these steps to build the WebAssembly filter:

### 1. Install the Emscripten SDK

If you don't have Emscripten installed:

```bash
# Clone and install Emscripten SDK
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
cd ..
```

### 2. Build the WebAssembly Module and Runtime Wrapper

```bash
cd /path/to/bpftime/example/attach_implementation/benchmark/wasm_plugin

# Build the WebAssembly module and runtime wrapper
make
```

This will:
- Clone and build the Wasm3 WebAssembly runtime
- Compile the `url_filter.c` to WebAssembly (producing `url_filter.wasm`)
- Compile the `wasm_runtime.c` to a shared library (producing `libwasm_filter.so`)

## Running the WebAssembly Filter with NGINX

To run NGINX with the WebAssembly filter:

```bash
cd /path/to/bpftime/example/attach_implementation

# Start NGINX with the dynamic load module
DYNAMIC_LOAD_LIB_PATH="$(pwd)/benchmark/wasm_plugin/libwasm_filter.so" DYNAMIC_LOAD_URL_PREFIX="/aaaa" WASM_MODULE_PATH="$(pwd)/benchmark/wasm_plugin/url_filter.wasm" ./nginx_plugin_output/nginx -p $(pwd) -c benchmark/dynamic_load_module.conf
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

1. The WebAssembly module (`url_filter.wasm`) contains three exported functions:
   - `module_initialize`: Sets the URL prefix to accept and resets counters
   - `module_url_filter`: Checks if a URL starts with the accepted prefix
   - `module_get_counters`: Returns the number of accepted and rejected requests

2. The runtime wrapper (`libwasm_filter.so`) loads the WebAssembly module using Wasm3 and:
   - Provides the same API as the other filter implementations
   - Translates calls to the native API into calls to the WebAssembly module
   - Caches statistics for quick access

3. The `dynamic_load_plugin` in NGINX loads the runtime wrapper as a shared library and calls its functions to:
   - Initialize the filter with a URL prefix
   - Filter requests based on their URLs
   - Keep track of accepted and rejected requests

## Including in Benchmarks

To include the WebAssembly filter in the benchmark, you need to make sure the WebAssembly module and runtime wrapper are built before running the benchmark script:

```bash
# Build the WebAssembly filter
cd /path/to/bpftime/example/attach_implementation/benchmark/wasm_plugin
make

# Run the benchmark
cd /path/to/bpftime
python3 example/attach_implementation/benchmark/run_benchmark.py
```

The benchmark script automatically sets the necessary environment variables when testing the dynamic load module, so it will use the WebAssembly filter if available.

## Troubleshooting

### Missing Header Files

If you encounter errors about missing header files:

```
wasm_runtime.c:5:10: fatal error: wasm3.h: No such file or directory
```

Make sure you've run `make` from the `wasm_plugin` directory, as it will automatically clone and build Wasm3.

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