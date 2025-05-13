# Nginx Module Benchmarking

This directory contains benchmarking tools to compare the performance of four different Nginx configurations:

1. **Nginx with bpftime module** - Uses eBPF-based URL filtering
2. **Nginx with baseline C module** - Uses traditional C implementation with shared memory
3. **Nginx with dynamic load module** - Uses dynamic library loading at runtime
4. **Nginx without any module** - Baseline performance with no filtering

## Directory Structure

- `ebpf_controller/` - eBPF controller implementation
- `baseline_nginx_plugin/` - Traditional C module implementation
- `dynamic_load_plugin/` - Dynamic library loading implementation
- `wasm_plugin/` - WebAssembly-based filter implementation
- `run_benchmark.py` - Python script to run benchmarks
- `baseline_c_module.conf` - Nginx configuration for baseline C module
- `dynamic_load_module.conf` - Nginx configuration for dynamic load module
- `no_module.conf` - Nginx configuration without modules

## Prerequisites

- CMake (3.10+)
- C++ compiler with C++17 support
- Nginx
- wrk HTTP benchmarking tool
- Python 3.6+

## Building the Project

### 1. Build the main bpftime project

First, build the main bpftime project including the Nginx plugin.(See parent [README](../README.md) for details)

## Running Automatic Benchmarks

The `run_benchmark.py` script will automatically:

1. Start the controllers (before Nginx)
2. Start Nginx with each configuration
3. Run wrk benchmark against each
4. Collect and display results
5. Log all output to `benchlog.txt`

Run the benchmark with:

```bash
cd /path/to/bpftime
python3 example/attach_implementation/benchmark/run_benchmark.py
```

You can customize the benchmark with these options:

- `--duration`: Duration of each benchmark in seconds (default: 30)
- `--connections`: Number of connections to use (default: 400)
- `--threads`: Number of threads to use (default: 12)
- `--url-path`: URL path to test (default: "/aaaa")

For example, to run a shorter benchmark with fewer connections:

```bash
python3 example/attach_implementation/benchmark/run_benchmark.py --duration 10 --connections 100 --url-path /aaaa
```

### Benchmark Log

All output from the benchmark, including:

- Controller stdout/stderr
- Nginx stdout/stderr
- Detailed wrk benchmark results
- Error messages

are logged to the `benchlog.txt` file in the benchmark directory. This provides a complete record of the benchmark execution and can be useful for debugging if any issues occur.

The log entries are timestamped, making it easy to track the sequence of events during the benchmark.

## Interpreting Results

The benchmark script will output results like:

```
=== Benchmark Results Summary ===

Nginx without module:
  Requests/sec: 4536.86 ± 1027.29
  Latency (avg): 16.54ms
  Successful iterations: 20

Nginx with baseline C module:
  Requests/sec: 4559.84 ± 1000.22
  Latency (avg): 16.39ms
  Successful iterations: 20

Nginx with WebAssembly module:
  Requests/sec: 4008.00 ± 797.06
  Latency (avg): 18.11ms
  Successful iterations: 20

Nginx with LuaJIT module:
  Requests/sec: 3982.76 ± 644.57
  Latency (avg): 17.94ms
  Successful iterations: 20

Nginx with bpftime module:
  Requests/sec: 4461.00 ± 1031.08
  Latency (avg): 16.68ms
  Successful iterations: 20

Nginx with RLBox NoOp module:
  Requests/sec: 4148.58 ± 687.99
  Latency (avg): 17.25ms
  Successful iterations: 20

Nginx with ERIM-protected module:
  Requests/sec: 4024.11 ± 784.07
  Latency (avg): 18.80ms
  Successful iterations: 20

Overhead Comparisons:
  Compared to no module:
    Baseline C: -0.51%
    WebAssembly: 11.66%
    LuaJIT: 12.21%
    BPFtime: 1.67%
    RLBox NoOp: 8.56%
    ERIM: 11.30%
  Compared to baseline C module:
    WebAssembly: 12.10%
    LuaJIT: 12.66%
    BPFtime: 2.17%
    RLBox NoOp: 9.02%
    ERIM: 11.75%
  Compared to WebAssembly module:
    BPFtime: -11.30%
    ERIM: -0.40%
  Compared to LuaJIT module:
    BPFtime: -12.01%
    ERIM: -1.04%
  Compared to RLBox NoOp module:
    BPFtime: -7.53%
    ERIM: 3.00%
  Compared to ERIM-protected module:
    BPFtime: -10.86%
JSON results saved to: /home/yunwei37/bpftime/example/attach_implementation/benchmark/benchmark_results_20250512_173550.json
```

See the [example_benchmark_results.json](example_benchmark_results.json) for more details.

Key metrics to consider:

- **Requests/sec**: Higher is better
- **Latency**: Lower is better
- **Overhead percentages**: The performance cost of each filtering approach

## Manual Testing

To manually test the correctness of each implementation, follow these steps:

### Testing the Baseline C Module

> **IMPORTANT**: The controller must be started BEFORE the Nginx instance for the baseline C module. This is because the controller creates the shared memory that Nginx connects to.

1. Start the baseline controller first:

   ```bash
   cd /path/to/bpftime
   build/example/attach_implementation/benchmark/baseline_nginx_plugin/nginx_baseline_controller /aaaa
   ```

   This configures the filter to accept URLs starting with "/aaaa" and reject others.

2. In a new terminal, start Nginx with the baseline module:

   ```bash
   cd /path/to/bpftime/example/attach_implementation/
   nginx_plugin_output/nginx -p $(pwd) -c benchmark/baseline_c_module.conf
   ```

3. Test with curl:

   ```bash
   # This should succeed (HTTP 200)
   curl http://localhost:9025/aaaa

   # This should fail (HTTP 403 Forbidden)
   curl http://localhost:9025/forbidden_path
   ```

4. Check the controller's output to see the accepted and rejected counts.

### Testing the eBPF/bpftime Module

1. Start the eBPF controller:

   ```bash
   cd /path/to/bpftime/build
   ./example/attach_implementation/benchmark/ebpf_controller/nginx_benchmark_ebpf_controller /aaaa
   ```

2. In a new terminal, start Nginx with the bpftime module:

   ```bash
   cd /path/to/bpftime/example/attach_implementation
   ./nginx_plugin_output/nginx -p $(pwd) -c nginx.conf
   ```

3. Test with curl:

   ```bash
   # This should succeed (HTTP 200)
   curl http://localhost:9023/aaaa

   # This should fail (HTTP 403 Forbidden)
   curl http://localhost:9023/forbidden_path
   ```

4. Check the controller's output to see processed requests.

### Testing the Dynamic Load Module

The dynamic load module doesn't require a separate controller as it loads the filter implementation directly from a shared library.

This will be used for wasm module and lua module.

It is configured through environment variables:

1. Build the filter implementation library first:

   ```bash
   cd /path/to/bpftime/example/attach_implementation/benchmark/dynamic_load_plugin/dynamic_tests
   make
   ```

2. Start Nginx with the dynamic load module:

   ```bash
   cd /path/to/bpftime/example/attach_implementation
   DYNAMIC_LOAD_LIB_PATH="/home/yunwei37/bpftime/example/attach_implementation/benchmark/dynamic_load_plugin/dynamic_tests/libfilter_impl.so"  DYNAMIC_LOAD_URL_PREFIX="/aaaa" nginx_plugin_output/nginx -p $(pwd) -c benchmark/dynamic_load_module.conf
   ```

3. Test with curl:

   ```bash
   # This should succeed (HTTP 200)
   curl http://localhost:9026/aaaa

   # This should fail (HTTP 403 Forbidden)
   curl http://localhost:9026/forbidden_path
   ```

5. The filter library logs accepted/rejected requests internally, which can be retrieved from the nginx logs.

### Testing the WebAssembly Filter

The WebAssembly filter leverages the dynamic load module infrastructure but uses a WebAssembly runtime to execute the filter logic.

1. Build the WebAssembly module and runtime wrapper:

   ```bash
   cd /path/to/bpftime/example/attach_implementation/benchmark/wasm_plugin
   make
   ```

   For more details, see the [WebAssembly Plugin README](wasm_plugin/README.md).

2. Start Nginx with the WebAssembly filter:

   ```bash
   cd /path/to/bpftime/example/attach_implementation
   
   # Set the necessary environment variables
   export DYNAMIC_LOAD_LIB_PATH="$(pwd)/benchmark/wasm_plugin/libwasm_filter.so"
   export DYNAMIC_LOAD_URL_PREFIX="/aaaa"
   export WASM_MODULE_PATH="$(pwd)/benchmark/wasm_plugin/url_filter.wasm"
   
   # Start Nginx with the dynamic load module
   ./nginx_plugin_output/nginx -p $(pwd) -c benchmark/dynamic_load_module.conf
   ```

3. Test with curl:

   ```bash
   # This should succeed (HTTP 200)
   curl http://localhost:9026/aaaa

   # This should fail (HTTP 403 Forbidden)
   curl http://localhost:9026/forbidden_path
   ```

## Notes

- The performance comparison focuses on the overhead introduced by each module implementation
- All implementations use the same string comparison logic for URL filtering
- The baseline C module uses shared memory to share data with its controller
- The bpftime module uses eBPF to implement the filtering logic
- The dynamic load module loads a filter implementation shared library at runtime
- The WebAssembly filter compiles the filter logic to WebAssembly and executes it in a WebAssembly runtime

## Troubleshooting

### Nginx fails to start with "Failed to open shared memory" error

- Make sure the controller is running BEFORE starting Nginx
- The controller creates the shared memory that Nginx connects to
- If Nginx is started first, it will fail to find the shared memory

### "Failed to open shared memory" error

- Ensure no previous instances of the controller or Nginx are running
- You may need to manually clean up shared memory if a previous run crashed:

  ```bash
  rm /dev/shm/baseline_nginx_filter_shm
  ```

### Controller or Nginx not starting

- Check port conflicts (9023, 9024, 9025)
- Ensure you have permissions to bind to these ports
- Look for errors in the nginx error log or in benchlog.txt  
