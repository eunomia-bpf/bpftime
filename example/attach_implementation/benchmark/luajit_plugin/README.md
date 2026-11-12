# LuaJIT URL Filter Plugin

This directory contains a LuaJIT-based implementation of the URL filter that can be loaded by the `dynamic_load_plugin` in NGINX. The implementation consists of two main components:

1. **Lua Script** (`url_filter.lua`): This contains the actual URL filtering logic written in Lua.
2. **Runtime Wrapper** (`lua_runtime.c`): This is a native C library that loads and executes the Lua script, implementing the same API as the other filter libraries.

## Prerequisites

To build and run the LuaJIT filter, you need:

- C/C++ compiler (GCC or Clang)
- Git (for cloning repositories if building LuaJIT from source)
- Either:
  - Pre-installed LuaJIT and its development headers
  - Or build environment to compile LuaJIT from source (automatically handled by the Makefile)

## Building the LuaJIT Filter

Follow these steps to build the LuaJIT filter:

### 1. Install or Build LuaJIT

The Makefile will attempt to:
1. Find an existing LuaJIT installation using pkg-config
2. Check common system paths for LuaJIT headers and libraries
3. If not found, automatically download and build LuaJIT from source

```bash
# Build the LuaJIT filter (will handle dependencies as needed)
cd /path/to/bpftime/example/attach_implementation/benchmark/luajit_plugin
make
```

If you prefer to install LuaJIT explicitly via your package manager:

```bash
# On Debian/Ubuntu
sudo apt-get update && sudo apt-get install -y luajit libluajit-5.1-dev

# On CentOS/RHEL
sudo yum install -y luajit luajit-devel

# Then build the filter
make
```

## Running the LuaJIT Filter with NGINX

To run NGINX with the LuaJIT filter:

```bash
cd /path/to/bpftime/example/attach_implementation

# Start NGINX with the dynamic load module
export DYNAMIC_LOAD_LIB_PATH="$(pwd)/benchmark/luajit_plugin/liblua_filter.so"
export DYNAMIC_LOAD_URL_PREFIX="/aaaa"
export LUA_MODULE_PATH="$(pwd)/benchmark/luajit_plugin/url_filter.lua"
./nginx_plugin_output/nginx -p $(pwd) -c benchmark/dynamic_load_module.conf
```

## Testing the LuaJIT Filter

Once NGINX is running with the LuaJIT filter, you can test it using curl:

```bash
# This should succeed (HTTP 200)
curl http://localhost:9026/aaaa

# This should fail (HTTP 403 Forbidden)
curl http://localhost:9026/forbidden_path
```

## How It Works

1. The Lua script (`url_filter.lua`) contains the following exported functions:
   - `initialize`: Sets the URL prefix to accept and resets counters
   - `url_filter`: Checks if a URL starts with the accepted prefix
   - `get_counters`: Returns the number of accepted and rejected requests
   - `set_buffer`: Sets data in a buffer
   - `get_buffer`: Gets data from a buffer

2. The runtime wrapper (`liblua_filter.so`):
   - Uses LuaJIT to execute the Lua script
   - Loads the Lua script from a file specified by the `LUA_MODULE_PATH` environment variable
   - Provides the same API as other filter implementations
   - Translates calls between the native API and the Lua script

3. The `dynamic_load_plugin` in NGINX loads the runtime wrapper as a shared library and calls its functions to:
   - Initialize the filter with a URL prefix
   - Filter requests based on their URLs
   - Keep track of accepted and rejected requests

## Including in Benchmarks

The LuaJIT filter can be included in the benchmark script. To run the benchmark with the LuaJIT filter:

```bash
# Build the LuaJIT filter first
cd /path/to/bpftime/example/attach_implementation/benchmark/luajit_plugin
make

# Run the benchmark
cd /path/to/bpftime
python3 example/attach_implementation/benchmark/run_benchmark.py
```

The benchmark script will automatically build and use the LuaJIT filter when testing.

## Performance Characteristics

LuaJIT is known for its high performance due to its Just-In-Time compilation capabilities:

- **JIT Compilation**: LuaJIT compiles hot code paths to native machine code
- **Fast FFI**: The Foreign Function Interface allows for low-overhead calls to C functions
- **Minimal Overhead**: The implementation focuses on reducing overhead between Lua and C code
- **Small Memory Footprint**: LuaJIT has a small memory footprint compared to many other scripting runtimes

For URL filtering workloads, LuaJIT typically provides performance close to native C implementations while offering the flexibility of a dynamic language.

## Troubleshooting

### LuaJIT Not Found During Build

If you see errors about LuaJIT headers not being found, the Makefile will automatically attempt to download and build LuaJIT locally. Ensure you have git installed and can access GitHub.

If you prefer to install LuaJIT manually:

```bash
# On Debian/Ubuntu
sudo apt-get update && sudo apt-get install -y luajit libluajit-5.1-dev

# On CentOS/RHEL
sudo yum install -y luajit luajit-devel
```

### Lua Script Not Found at Runtime

If you get errors about the Lua script not being found:

```
Failed to load Lua module: cannot open url_filter.lua: No such file or directory
```

Make sure to set the `LUA_MODULE_PATH` environment variable to the absolute path of the Lua script.

```bash
export LUA_MODULE_PATH="/absolute/path/to/url_filter.lua"
```

### Shared Library Not Found

If NGINX can't load the shared library:

```
Failed to load shared library: cannot open shared object file: No such file or directory
```

Make sure the `DYNAMIC_LOAD_LIB_PATH` environment variable is set to the absolute path of the shared library.

```bash
export DYNAMIC_LOAD_LIB_PATH="/absolute/path/to/liblua_filter.so"
```

### Library Dependencies Missing

If you encounter errors about missing LuaJIT libraries at runtime:

```
liblua_filter.so: cannot open shared object file: No such file or directory
```

Make sure the LuaJIT library is in your library path. If you built LuaJIT from source, you may need to:

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/path/to/luajit-local/src"
``` 