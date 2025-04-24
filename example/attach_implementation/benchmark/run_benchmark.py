#!/usr/bin/env python3
"""
Benchmark script for comparing different nginx configurations:
1. With bpftime module (eBPF-based filtering)
2. With baseline C module (direct C implementation)
3. With WebAssembly module (WASM-based filtering)
4. With LuaJIT module (Lua-based filtering)
5. With ERIM-protected module (MPK-based isolation)
6. Without any module (baseline performance)

This script will:
- Start each nginx configuration
- Run wrk benchmarks against each
- Collect and display results
- Run multiple iterations if requested and average the results
- Output results to a JSON file
"""

import os
import subprocess
import time
import signal
import argparse
import sys
import datetime
import json
from pathlib import Path
from statistics import mean, stdev

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

def run_benchmark_iteration(args):
    """Run a single iteration of the benchmark suite"""
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
    
    # Test with RLBox module
    log_message("\n=== Testing nginx with RLBox module ===")
    # First make sure the RLBox module is built
    try:
        # Build the appropriate variant based on command-line argument
        if args.rlbox_variant == "wasm2c":
            build_cmd = ["make", "wasm2c", "-C", str(SCRIPT_DIR / "rlbox_plugin")]
            log_message(f"Building RLBox wasm2c sandbox with command: {' '.join(build_cmd)}")
        else:
            build_cmd = ["make", "-C", str(SCRIPT_DIR / "rlbox_plugin")]
            log_message(f"Building RLBox noop sandbox with command: {' '.join(build_cmd)}")
        
        subprocess.run(build_cmd, check=True)
    except subprocess.CalledProcessError as e:
        log_message(f"Failed to build RLBox module: {e}")
        log_message("Skipping RLBox benchmark")
    else:
        # Set up environment variables for the RLBox module
        rlbox_env = os.environ.copy()
        
        if args.rlbox_variant == "wasm2c":
            rlbox_lib_path = str(SCRIPT_DIR / "rlbox_plugin" / "libfilter_rlbox_wasm2c.so")
            log_message("Using wasm2c sandbox (production) for RLBox")
        else:
            rlbox_lib_path = str(SCRIPT_DIR / "rlbox_plugin" / "libfilter_rlbox.so")
            log_message("Using noop sandbox (development) for RLBox")
            
        rlbox_env["DYNAMIC_LOAD_LIB_PATH"] = rlbox_lib_path
        rlbox_env["DYNAMIC_LOAD_URL_PREFIX"] = args.url_path
        log_message(f"Setting DYNAMIC_LOAD_LIB_PATH={rlbox_lib_path}")
        log_message(f"Setting DYNAMIC_LOAD_URL_PREFIX={args.url_path}")
        
        # Use the same dynamic_load_module.conf and port - we're running these tests sequentially
        nginx_process = start_nginx(NGINX_BIN, DYNAMIC_LOAD_CONF, PARENT_DIR, env=rlbox_env)
        if nginx_process:
            url = f"http://127.0.0.1:{WASM_PORT}{args.url_path}"
            time.sleep(1)
            output = run_wrk_benchmark(url, args.duration, args.connections, args.threads)
            
            # Use different result keys for the variants
            if args.rlbox_variant == "wasm2c":
                results['rlbox_wasm2c'] = parse_wrk_output(output)
                collect_process_output(nginx_process, "RLBox Wasm2c Nginx")
            else:
                results['rlbox'] = parse_wrk_output(output)
                collect_process_output(nginx_process, "RLBox NoOp Nginx")
                
            stop_nginx(nginx_process, PARENT_DIR)
    
    # Test with ERIM-protected module
    log_message("\n=== Testing nginx with ERIM-protected module ===")
    # First make sure the ERIM module is built
    try:
        build_cmd = ["make", "-C", str(SCRIPT_DIR / "erim_plugin")]
        log_message(f"Building ERIM-protected module with command: {' '.join(build_cmd)}")
        subprocess.run(build_cmd, check=True)
    except subprocess.CalledProcessError as e:
        log_message(f"Failed to build ERIM-protected module: {e}")
        log_message("Skipping ERIM benchmark")
    else:
        # Set up environment variables for the ERIM module
        erim_env = os.environ.copy()
        erim_lib_path = str(SCRIPT_DIR / "erim_plugin" / "liberim_filter.so")
        erim_env["DYNAMIC_LOAD_LIB_PATH"] = erim_lib_path
        erim_env["DYNAMIC_LOAD_URL_PREFIX"] = args.url_path
        log_message(f"Setting DYNAMIC_LOAD_LIB_PATH={erim_lib_path}")
        log_message(f"Setting DYNAMIC_LOAD_URL_PREFIX={args.url_path}")
        
        # Use the same dynamic_load_module.conf and port - we're running these tests sequentially
        nginx_process = start_nginx(NGINX_BIN, DYNAMIC_LOAD_CONF, PARENT_DIR, env=erim_env)
        if nginx_process:
            url = f"http://127.0.0.1:{WASM_PORT}{args.url_path}"
            time.sleep(1)
            output = run_wrk_benchmark(url, args.duration, args.connections, args.threads)
            results['erim'] = parse_wrk_output(output)
            collect_process_output(nginx_process, "ERIM-protected Nginx")
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
    
    return results

def calculate_averages(all_results):
    """Calculate average metrics from multiple benchmark iterations"""
    avg_results = {}
    std_dev = {}
    
    # Initialize results structure
    for module in ['no_module', 'baseline', 'wasm', 'lua', 'bpftime', 'rlbox', 'rlbox_wasm2c', 'erim']:
        if any(module in results for results in all_results):
            avg_results[module] = {'rps': 0, 'latency_avg': '0ms', 'iterations': 0}
            std_dev[module] = {'rps': 0}
    
    # Collect all RPS values by module
    rps_by_module = {module: [] for module in avg_results}
    latency_by_module = {module: [] for module in avg_results}
    
    # Gather metrics from all iterations
    for result in all_results:
        for module, metrics in result.items():
            if module in avg_results:
                if 'rps' in metrics:
                    rps_by_module[module].append(metrics['rps'])
                if 'latency_avg' in metrics:
                    # Convert latency to float for averaging (remove 'ms' suffix)
                    try:
                        lat_value = float(metrics['latency_avg'].rstrip('ms'))
                        latency_by_module[module].append(lat_value)
                    except ValueError:
                        # If cannot convert to float, skip this latency value
                        pass
    
    # Calculate averages and standard deviations
    for module in avg_results:
        if rps_by_module[module]:
            avg_results[module]['rps'] = mean(rps_by_module[module])
            avg_results[module]['iterations'] = len(rps_by_module[module])
            if len(rps_by_module[module]) > 1:
                std_dev[module]['rps'] = stdev(rps_by_module[module])
            else:
                std_dev[module]['rps'] = 0
                
        if latency_by_module[module]:
            avg_latency = mean(latency_by_module[module])
            avg_results[module]['latency_avg'] = f"{avg_latency:.2f}ms"
            if len(latency_by_module[module]) > 1:
                std_dev[module]['latency_avg'] = f"{stdev(latency_by_module[module]):.2f}ms"
            else:
                std_dev[module]['latency_avg'] = "0ms"
    
    return avg_results, std_dev, rps_by_module

def calculate_overheads(avg_results):
    """Calculate the performance overhead between different modules"""
    overheads = {}
    
    # No module as baseline
    if 'no_module' in avg_results and avg_results['no_module']['rps'] > 0:
        no_module_rps = avg_results['no_module']['rps']
        
        for module in ['baseline', 'wasm', 'lua', 'bpftime', 'rlbox', 'rlbox_wasm2c', 'erim']:
            if module in avg_results and avg_results[module]['rps'] > 0:
                overhead = (1 - avg_results[module]['rps'] / no_module_rps) * 100
                overheads[f"{module}_vs_no_module"] = f"{overhead:.2f}%"
    
    # C baseline module comparisons
    if 'baseline' in avg_results and avg_results['baseline']['rps'] > 0:
        baseline_rps = avg_results['baseline']['rps']
        
        for module in ['wasm', 'lua', 'bpftime', 'rlbox', 'rlbox_wasm2c', 'erim']:
            if module in avg_results and avg_results[module]['rps'] > 0:
                overhead = (1 - avg_results[module]['rps'] / baseline_rps) * 100
                overheads[f"{module}_vs_baseline"] = f"{overhead:.2f}%"
    
    # WebAssembly module comparisons
    if 'wasm' in avg_results and avg_results['wasm']['rps'] > 0:
        wasm_rps = avg_results['wasm']['rps']
        
        if 'bpftime' in avg_results and avg_results['bpftime']['rps'] > 0:
            overhead = (1 - avg_results['bpftime']['rps'] / wasm_rps) * 100
            overheads["bpftime_vs_wasm"] = f"{overhead:.2f}%"
            
        if 'erim' in avg_results and avg_results['erim']['rps'] > 0:
            overhead = (1 - avg_results['erim']['rps'] / wasm_rps) * 100
            overheads["erim_vs_wasm"] = f"{overhead:.2f}%"
    
    # LuaJIT module comparisons
    if 'lua' in avg_results and avg_results['lua']['rps'] > 0:
        lua_rps = avg_results['lua']['rps']
        
        if 'bpftime' in avg_results and avg_results['bpftime']['rps'] > 0:
            overhead = (1 - avg_results['bpftime']['rps'] / lua_rps) * 100
            overheads["bpftime_vs_lua"] = f"{overhead:.2f}%"
            
        if 'erim' in avg_results and avg_results['erim']['rps'] > 0:
            overhead = (1 - avg_results['erim']['rps'] / lua_rps) * 100
            overheads["erim_vs_lua"] = f"{overhead:.2f}%"
    
    # RLBox module comparisons (noop variant)
    if 'rlbox' in avg_results and avg_results['rlbox']['rps'] > 0:
        rlbox_rps = avg_results['rlbox']['rps']
        
        if 'bpftime' in avg_results and avg_results['bpftime']['rps'] > 0:
            overhead = (1 - avg_results['bpftime']['rps'] / rlbox_rps) * 100
            overheads["bpftime_vs_rlbox"] = f"{overhead:.2f}%"
            
        if 'erim' in avg_results and avg_results['erim']['rps'] > 0:
            overhead = (1 - avg_results['erim']['rps'] / rlbox_rps) * 100
            overheads["erim_vs_rlbox"] = f"{overhead:.2f}%"
            
    # RLBox module comparisons (wasm2c variant)
    if 'rlbox_wasm2c' in avg_results and avg_results['rlbox_wasm2c']['rps'] > 0:
        rlbox_wasm2c_rps = avg_results['rlbox_wasm2c']['rps']
        
        if 'bpftime' in avg_results and avg_results['bpftime']['rps'] > 0:
            overhead = (1 - avg_results['bpftime']['rps'] / rlbox_wasm2c_rps) * 100
            overheads["bpftime_vs_rlbox_wasm2c"] = f"{overhead:.2f}%"
            
        if 'erim' in avg_results and avg_results['erim']['rps'] > 0:
            overhead = (1 - avg_results['erim']['rps'] / rlbox_wasm2c_rps) * 100
            overheads["erim_vs_rlbox_wasm2c"] = f"{overhead:.2f}%"
            
        # Compare RLBox wasm2c with WebAssembly module
        if 'wasm' in avg_results and avg_results['wasm']['rps'] > 0:
            overhead = (1 - avg_results['rlbox_wasm2c']['rps'] / avg_results['wasm']['rps']) * 100
            overheads["rlbox_wasm2c_vs_wasm"] = f"{overhead:.2f}%"
            
        # Compare RLBox NoOp with RLBox wasm2c
        if 'rlbox' in avg_results and avg_results['rlbox']['rps'] > 0:
            overhead = (1 - avg_results['rlbox_wasm2c']['rps'] / avg_results['rlbox']['rps']) * 100
            overheads["rlbox_wasm2c_vs_rlbox"] = f"{overhead:.2f}%"
    
    # ERIM module comparisons
    if 'erim' in avg_results and avg_results['erim']['rps'] > 0:
        erim_rps = avg_results['erim']['rps']
        
        if 'bpftime' in avg_results and avg_results['bpftime']['rps'] > 0:
            overhead = (1 - avg_results['bpftime']['rps'] / erim_rps) * 100
            overheads["bpftime_vs_erim"] = f"{overhead:.2f}%"
    
    return overheads

def log_json_results(avg_results, std_dev, overheads, all_iterations, args):
    """Create and save JSON results file"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file = str(SCRIPT_DIR / f"benchmark_results_{timestamp}.json")
    
    # Prepare JSON structure
    json_data = {
        "benchmark_info": {
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "duration": args.duration,
            "connections": args.connections,
            "threads": args.threads,
            "url_path": args.url_path,
            "iterations": args.iterations
        },
        "average_results": avg_results,
        "standard_deviation": std_dev,
        "overheads": overheads,
    }
    
    # Add raw iteration data
    if args.save_all_iterations:
        json_data["all_iterations"] = all_iterations
    
    # Write to file
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    log_message(f"JSON results saved to: {json_file}")
    return json_file

def print_results_summary(avg_results, std_dev, overheads):
    """Print a summary of the benchmark results to the console and log"""
    summary = "\n\n=== Benchmark Results Summary ==="
    
    # Print module results
    for module in ['no_module', 'baseline', 'wasm', 'lua', 'bpftime', 'rlbox', 'rlbox_wasm2c', 'erim']:
        if module in avg_results:
            module_name = {
                'no_module': 'Nginx without module',
                'baseline': 'Nginx with baseline C module',
                'wasm': 'Nginx with WebAssembly module',
                'lua': 'Nginx with LuaJIT module',
                'bpftime': 'Nginx with bpftime module',
                'rlbox': 'Nginx with RLBox NoOp module',
                'rlbox_wasm2c': 'Nginx with RLBox Wasm2c module',
                'erim': 'Nginx with ERIM-protected module'
            }.get(module, module)
            
            summary += f"\n\n{module_name}:"
            summary += f"\n  Requests/sec: {avg_results[module]['rps']:.2f} ± {std_dev[module]['rps']:.2f}"
            summary += f"\n  Latency (avg): {avg_results[module]['latency_avg']}"
            summary += f"\n  Successful iterations: {avg_results[module]['iterations']}"
    
    # Print overhead comparisons
    if overheads:
        summary += "\n\nOverhead Comparisons:"
        
        # Group overheads by base module
        for base in ['no_module', 'baseline', 'wasm', 'lua', 'rlbox', 'rlbox_wasm2c', 'erim']:
            base_name = {
                'no_module': 'no module',
                'baseline': 'baseline C module',
                'wasm': 'WebAssembly module', 
                'lua': 'LuaJIT module',
                'rlbox': 'RLBox NoOp module',
                'rlbox_wasm2c': 'RLBox Wasm2c module',
                'erim': 'ERIM-protected module'
            }.get(base, base)
            
            relevant_overheads = {k: v for k, v in overheads.items() if f"_vs_{base}" in k}
            
            if relevant_overheads:
                summary += f"\n  Compared to {base_name}:"
                for k, v in relevant_overheads.items():
                    module = k.split('_vs_')[0]
                    module_name = {
                        'baseline': 'Baseline C',
                        'wasm': 'WebAssembly',
                        'lua': 'LuaJIT',
                        'bpftime': 'BPFtime',
                        'rlbox': 'RLBox NoOp',
                        'rlbox_wasm2c': 'RLBox Wasm2c',
                        'erim': 'ERIM'
                    }.get(module, module)
                    summary += f"\n    {module_name}: {v}"
    
    log_message(summary)
    return summary

def main():
    parser = argparse.ArgumentParser(description="Run nginx benchmarks with different configurations")
    parser.add_argument("--duration", type=int, default=60, help="Duration of each benchmark in seconds")
    parser.add_argument("--connections", type=int, default=2000, help="Number of connections to use")
    parser.add_argument("--threads", type=int, default=6, help="Number of threads to use")
    parser.add_argument("--url-path", type=str, default="/aaaa", help="URL path to test")
    parser.add_argument("--iterations", type=int, default=1, help="Number of benchmark iterations to run")
    parser.add_argument("--save-all-iterations", action="store_true", help="Save all iteration data in the JSON output")
    parser.add_argument("--json-output", type=str, help="Custom path for JSON output file")
    parser.add_argument("--rlbox-variant", type=str, choices=["noop", "wasm2c"], default="noop", 
                       help="RLBox sandbox variant to use (noop for development, wasm2c for production)")
    args = parser.parse_args()
    
    # Initialize or clear the log file
    setup_log(BENCHMARK_LOG)
    with open(BENCHMARK_LOG, 'w') as f:
        f.write(f"=== Benchmark started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"Duration: {args.duration}s, Connections: {args.connections}, Threads: {args.threads}, URL: {args.url_path}\n")
        f.write(f"Iterations: {args.iterations}\n\n")
    
    check_prerequisites(["wrk", "nginx"], NGINX_BIN)
    
    all_results = []  # Store results from each iteration
    
    try:
        # Run the benchmark iterations
        for i in range(args.iterations):
            log_message(f"\n\n=== Running benchmark iteration {i+1}/{args.iterations} ===\n")
            iteration_results = run_benchmark_iteration(args)
            all_results.append(iteration_results)
            
            # Print intermediate results after each iteration
            log_message(f"\n=== Results from iteration {i+1}/{args.iterations} ===")
            for module, metrics in iteration_results.items():
                if metrics and 'rps' in metrics:
                    log_message(f"{module}: {metrics['rps']:.2f} req/s, {metrics['latency_avg']} latency")
        
        # Calculate averages and standard deviations
        avg_results, std_dev, raw_data = calculate_averages(all_results)
        
        # Calculate performance overheads
        overheads = calculate_overheads(avg_results)
        
        # Print results summary
        print_results_summary(avg_results, std_dev, overheads)
        
        # Save results to JSON
        json_file = log_json_results(avg_results, std_dev, overheads, raw_data, args)
        
        log_message(f"\n=== Benchmark completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        log_message(f"Full log available at: {BENCHMARK_LOG}")
        log_message(f"Results summary available at: {json_file}")
        
    except KeyboardInterrupt:
        log_message("\nBenchmark interrupted by user")
        sys.exit(0)
    except Exception as e:
        log_message(f"\nUnexpected error: {str(e)}")
        import traceback
        log_message(traceback.format_exc(), also_print=False)
        sys.exit(1)

if __name__ == "__main__":
    main() 