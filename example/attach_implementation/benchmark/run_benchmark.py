#!/usr/bin/env python3
"""
Benchmark script for comparing different nginx configurations:
1. With bpftime module (eBPF-based filtering)
2. With baseline C module (direct C implementation)
3. With WebAssembly module (WASM-based filtering)
4. With LuaJIT module (Lua-based filtering)
5. Without any module (baseline performance)

This script will:
- Start each nginx configuration
- Run wrk benchmarks against each
- Collect and display results
"""

import os
import subprocess
import time
import signal
import argparse
import sys
import datetime
from pathlib import Path

# Import utility functions
from utils import (
    setup_log, log_message, check_prerequisites, start_nginx, stop_nginx,
    collect_process_output, run_wrk_benchmark, parse_wrk_output,
    start_controller, stop_controller
)

# Get the directory of this script
SCRIPT_DIR = Path(__file__).parent.absolute()
# Get the parent directory (attach_implementation)
PARENT_DIR = SCRIPT_DIR.parent

# Configuration
NGINX_BIN = str(PARENT_DIR / "nginx_plugin_output" / "nginx")
BPFTIME_CONF = str(PARENT_DIR / "nginx.conf")
BASELINE_CONF = str(SCRIPT_DIR / "baseline_c_module.conf")
DYNAMIC_LOAD_CONF = str(SCRIPT_DIR / "dynamic_load_module.conf")
NO_MODULE_CONF = str(SCRIPT_DIR / "no_module.conf")

# Ports
BPFTIME_PORT = 9023
BASELINE_PORT = 9025
WASM_PORT = 9026  # Using the same port for dynamic load plugins
NO_MODULE_PORT = 9024

# URLs for testing
TEST_URL_PREFIX = "/aaaa"  # This should match what's allowed in your eBPF program

# Log file
BENCHMARK_LOG = str(SCRIPT_DIR / "benchlog.txt")

def start_bpftime_controller(prefix):
    """Start the bpftime controller with the specified prefix"""
    controller_path = PARENT_DIR.parent.parent / "build/example/attach_implementation/benchmark/ebpf_controller/nginx_benchmark_ebpf_controller"
    return start_controller(controller_path, prefix, "bpftime")

def start_baseline_controller(prefix):
    """Start the baseline controller with the specified prefix"""
    controller_path = PARENT_DIR.parent.parent / "build/example/attach_implementation/benchmark/baseline_nginx_plugin/nginx_baseline_controller"
    if not controller_path.exists():
        log_message(f"Error: Baseline controller not found at {controller_path}")
        log_message("Make sure to build the baseline controller first")
        return None
    
    return start_controller(controller_path, prefix, "baseline")

def main():
    parser = argparse.ArgumentParser(description="Run nginx benchmarks with different configurations")
    parser.add_argument("--duration", type=int, default=30, help="Duration of each benchmark in seconds")
    parser.add_argument("--connections", type=int, default=4000, help="Number of connections to use")
    parser.add_argument("--threads", type=int, default=12, help="Number of threads to use")
    parser.add_argument("--url-path", type=str, default="/aaaa", help="URL path to test")
    args = parser.parse_args()
    
    # Initialize or clear the log file
    setup_log(BENCHMARK_LOG)
    with open(BENCHMARK_LOG, 'w') as f:
        f.write(f"=== Benchmark started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"Duration: {args.duration}s, Connections: {args.connections}, Threads: {args.threads}, URL: {args.url_path}\n\n")
    
    check_prerequisites(["wrk", "nginx"], NGINX_BIN)
    
    results = {}
    
    # Test with no module
    log_message("\n=== Testing nginx without any module ===")
    nginx_process = start_nginx(NGINX_BIN, NO_MODULE_CONF, PARENT_DIR)
    if nginx_process:
        url = f"http://127.0.0.1:{NO_MODULE_PORT}{args.url_path}"
        output = run_wrk_benchmark(url, args.duration, args.connections, args.threads)
        results['no_module'] = parse_wrk_output(output)
        collect_process_output(nginx_process, "No-module Nginx")
        stop_nginx(nginx_process, PARENT_DIR)
    
    # Test with baseline C module
    log_message("\n=== Testing nginx with baseline C module ===")
    baseline_controller_process = start_baseline_controller(args.url_path)
    if baseline_controller_process:
        nginx_process = start_nginx(NGINX_BIN, BASELINE_CONF, PARENT_DIR)
        if nginx_process:
            url = f"http://127.0.0.1:{BASELINE_PORT}{args.url_path}"
            # Collect some controller output before the benchmark
            collect_process_output(baseline_controller_process, "Baseline controller")
            time.sleep(1)
            output = run_wrk_benchmark(url, args.duration, args.connections, args.threads)
            results['baseline'] = parse_wrk_output(output)
            # Collect some controller output after the benchmark
            collect_process_output(baseline_controller_process, "Baseline controller")
            collect_process_output(nginx_process, "Baseline Nginx")
            stop_nginx(nginx_process, PARENT_DIR)
        stop_controller(baseline_controller_process, "Baseline")
    
    # Test with WebAssembly module
    log_message("\n=== Testing nginx with WebAssembly module ===")
    # First make sure the WebAssembly module and runtime wrapper are built
    try:
        build_cmd = ["make", "-C", str(SCRIPT_DIR / "wasm_plugin")]
        log_message(f"Building WebAssembly module with command: {' '.join(build_cmd)}")
        subprocess.run(build_cmd, check=True)
    except subprocess.CalledProcessError as e:
        log_message(f"Failed to build WebAssembly module: {e}")
    else:
        # Set up environment variables for the WebAssembly module
        wasm_env = os.environ.copy()
        lib_path = str(SCRIPT_DIR / "wasm_plugin" / "libwasm_filter.so")
        wasm_module_path = str(SCRIPT_DIR / "wasm_plugin" / "url_filter.wasm")
        wasm_env["DYNAMIC_LOAD_LIB_PATH"] = lib_path
        wasm_env["DYNAMIC_LOAD_URL_PREFIX"] = args.url_path
        wasm_env["WASM_MODULE_PATH"] = wasm_module_path
        log_message(f"Setting DYNAMIC_LOAD_LIB_PATH={lib_path}")
        log_message(f"Setting DYNAMIC_LOAD_URL_PREFIX={args.url_path}")
        log_message(f"Setting WASM_MODULE_PATH={wasm_module_path}")
        
        # Start nginx with WebAssembly module - no separate controller needed
        nginx_process = start_nginx(NGINX_BIN, DYNAMIC_LOAD_CONF, PARENT_DIR, env=wasm_env)
        if nginx_process:
            url = f"http://127.0.0.1:{WASM_PORT}{args.url_path}"
            time.sleep(1)
            output = run_wrk_benchmark(url, args.duration, args.connections, args.threads)
            results['wasm'] = parse_wrk_output(output)
            collect_process_output(nginx_process, "WebAssembly Nginx")
            stop_nginx(nginx_process, PARENT_DIR)
    
    # Test with LuaJIT module
    log_message("\n=== Testing nginx with LuaJIT module ===")
    # First make sure the LuaJIT module and runtime wrapper are built
    try:
        build_cmd = ["make", "-C", str(SCRIPT_DIR / "luajit_plugin")]
        log_message(f"Building LuaJIT module with command: {' '.join(build_cmd)}")
        subprocess.run(build_cmd, check=True)
    except subprocess.CalledProcessError as e:
        log_message(f"Failed to build LuaJIT module: {e}")
        log_message("Skipping LuaJIT benchmark")
    else:
        # Set up environment variables for the LuaJIT module
        lua_env = os.environ.copy()
        lua_lib_path = str(SCRIPT_DIR / "luajit_plugin" / "liblua_filter.so")
        lua_module_path = str(SCRIPT_DIR / "luajit_plugin" / "url_filter.lua")
        lua_env["DYNAMIC_LOAD_LIB_PATH"] = lua_lib_path
        lua_env["DYNAMIC_LOAD_URL_PREFIX"] = args.url_path
        lua_env["LUA_MODULE_PATH"] = lua_module_path
        log_message(f"Setting DYNAMIC_LOAD_LIB_PATH={lua_lib_path}")
        log_message(f"Setting DYNAMIC_LOAD_URL_PREFIX={args.url_path}")
        log_message(f"Setting LUA_MODULE_PATH={lua_module_path}")
        
        # Use the same dynamic_load_module.conf and port - we're running these tests sequentially
        nginx_process = start_nginx(NGINX_BIN, DYNAMIC_LOAD_CONF, PARENT_DIR, env=lua_env)
        if nginx_process:
            url = f"http://127.0.0.1:{WASM_PORT}{args.url_path}"
            time.sleep(1)
            output = run_wrk_benchmark(url, args.duration, args.connections, args.threads)
            results['lua'] = parse_wrk_output(output)
            collect_process_output(nginx_process, "LuaJIT Nginx")
            stop_nginx(nginx_process, PARENT_DIR)
    
    # Test with bpftime module
    log_message("\n=== Testing nginx with bpftime module ===")
    bpftime_controller_process = start_bpftime_controller(args.url_path)
    if bpftime_controller_process:
        nginx_process = start_nginx(NGINX_BIN, BPFTIME_CONF, PARENT_DIR)
        if nginx_process:
            url = f"http://127.0.0.1:{BPFTIME_PORT}{args.url_path}"
            # Collect some controller output before the benchmark
            collect_process_output(bpftime_controller_process, "BPFtime controller")
            time.sleep(1)
            output = run_wrk_benchmark(url, args.duration, args.connections, args.threads)
            results['bpftime'] = parse_wrk_output(output)
            # Collect some controller output after the benchmark
            collect_process_output(bpftime_controller_process, "BPFtime controller")
            collect_process_output(nginx_process, "BPFtime Nginx")
            stop_nginx(nginx_process, PARENT_DIR)
        stop_controller(bpftime_controller_process, "BPFtime")
    
    # Print and log results
    result_header = "\n=== Benchmark Results ==="
    log_message(result_header)
    
    if 'no_module' in results and results['no_module']:
        no_module_result = f"\nNginx without module:"
        no_module_result += f"\n  Requests/sec: {results['no_module']['rps']:.2f}"
        no_module_result += f"\n  Latency (avg): {results['no_module']['latency_avg']}"
        log_message(no_module_result)
    
    if 'baseline' in results and results['baseline']:
        baseline_result = f"\nNginx with baseline C module:"
        baseline_result += f"\n  Requests/sec: {results['baseline']['rps']:.2f}"
        baseline_result += f"\n  Latency (avg): {results['baseline']['latency_avg']}"
        log_message(baseline_result)
        
        # Calculate overhead compared to no module
        if 'no_module' in results and results['no_module']:
            overhead = (1 - results['baseline']['rps'] / results['no_module']['rps']) * 100
            log_message(f"  Overhead vs no module: {overhead:.2f}%")
    
    if 'wasm' in results and results['wasm']:
        wasm_result = f"\nNginx with WebAssembly module:"
        wasm_result += f"\n  Requests/sec: {results['wasm']['rps']:.2f}"
        wasm_result += f"\n  Latency (avg): {results['wasm']['latency_avg']}"
        log_message(wasm_result)
        
        # Calculate overhead compared to no module
        if 'no_module' in results and results['no_module']:
            overhead = (1 - results['wasm']['rps'] / results['no_module']['rps']) * 100
            log_message(f"  Overhead vs no module: {overhead:.2f}%")
        
        # Calculate overhead compared to baseline
        if 'baseline' in results and results['baseline']:
            overhead = (1 - results['wasm']['rps'] / results['baseline']['rps']) * 100
            log_message(f"  Overhead vs baseline C module: {overhead:.2f}%")

    if 'lua' in results and results['lua']:
        lua_result = f"\nNginx with LuaJIT module:"
        lua_result += f"\n  Requests/sec: {results['lua']['rps']:.2f}"
        lua_result += f"\n  Latency (avg): {results['lua']['latency_avg']}"
        log_message(lua_result)
        
        # Calculate overhead compared to no module
        if 'no_module' in results and results['no_module']:
            overhead = (1 - results['lua']['rps'] / results['no_module']['rps']) * 100
            log_message(f"  Overhead vs no module: {overhead:.2f}%")
        
        # Calculate overhead compared to baseline
        if 'baseline' in results and results['baseline']:
            overhead = (1 - results['lua']['rps'] / results['baseline']['rps']) * 100
            log_message(f"  Overhead vs baseline C module: {overhead:.2f}%")
        
        # Calculate overhead compared to WebAssembly module
        if 'wasm' in results and results['wasm']:
            overhead = (1 - results['lua']['rps'] / results['wasm']['rps']) * 100
            log_message(f"  Overhead vs WebAssembly module: {overhead:.2f}%")
    
    if 'bpftime' in results and results['bpftime']:
        bpftime_result = f"\nNginx with bpftime module:"
        bpftime_result += f"\n  Requests/sec: {results['bpftime']['rps']:.2f}"
        bpftime_result += f"\n  Latency (avg): {results['bpftime']['latency_avg']}"
        log_message(bpftime_result)
        
        # Calculate overhead compared to no module
        if 'no_module' in results and results['no_module']:
            overhead = (1 - results['bpftime']['rps'] / results['no_module']['rps']) * 100
            log_message(f"  Overhead vs no module: {overhead:.2f}%")
        
        # Calculate overhead compared to baseline
        if 'baseline' in results and results['baseline']:
            overhead = (1 - results['bpftime']['rps'] / results['baseline']['rps']) * 100
            log_message(f"  Overhead vs baseline C module: {overhead:.2f}%")
        
        # Calculate overhead compared to WebAssembly module
        if 'wasm' in results and results['wasm']:
            overhead = (1 - results['bpftime']['rps'] / results['wasm']['rps']) * 100
            log_message(f"  Overhead vs WebAssembly module: {overhead:.2f}%")
        
        # Calculate overhead compared to LuaJIT module
        if 'lua' in results and results['lua']:
            overhead = (1 - results['bpftime']['rps'] / results['lua']['rps']) * 100
            log_message(f"  Overhead vs LuaJIT module: {overhead:.2f}%")
    
    log_message(f"\n=== Benchmark completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    log_message(f"Full log available at: {BENCHMARK_LOG}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_message("\nBenchmark interrupted by user")
        sys.exit(0)
    except Exception as e:
        log_message(f"\nUnexpected error: {str(e)}")
        import traceback
        log_message(traceback.format_exc(), also_print=False)
        sys.exit(1) 