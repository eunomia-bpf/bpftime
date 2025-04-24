#!/usr/bin/env python3
"""
Benchmark script for comparing different nginx configurations:
1. With bpftime module (eBPF-based filtering)
2. With baseline C module (direct C implementation)
3. With WebAssembly module (WASM-based filtering)
4. Without any module (baseline performance)

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
import shutil
import datetime
import select
from pathlib import Path

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
WASM_PORT = 9026  # Using the same port as dynamic_load
NO_MODULE_PORT = 9024

# URLs for testing
TEST_URL_PREFIX = "/aaaa"  # This should match what's allowed in your eBPF program

# Log file
BENCHMARK_LOG = str(SCRIPT_DIR / "benchlog.txt")

def log_message(message, also_print=True):
    """Log a message to the benchmark log file and optionally print to console"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(BENCHMARK_LOG, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")
    if also_print:
        print(message)

def check_prerequisites():
    """Check if all required tools are available"""
    log_message("\n=== Checking prerequisites ===")
    required_tools = ["wrk", "nginx"]
    
    for tool in required_tools:
        if shutil.which(tool) is None:
            log_message(f"Error: {tool} is not installed or not in PATH")
            log_message(f"Please install {tool} and try again")
            sys.exit(1)
    
    if not os.path.exists(NGINX_BIN):
        log_message(f"Error: Nginx binary not found at {NGINX_BIN}")
        log_message("Please build the project first")
        sys.exit(1)
    
    log_message("All prerequisites are met")

def start_nginx(config_path, working_dir=PARENT_DIR, env=None):
    """Start nginx with the specified configuration"""
    cmd = [NGINX_BIN, "-p", str(working_dir), "-c", config_path]
    cmd_str = ' '.join(cmd)
    log_message(f"Starting nginx with command: {cmd_str}")
    
    # If environment variables are provided, log them
    if env:
        for key, value in env.items():
            log_message(f"Environment variable: {key}={value}")
    
    # Start nginx as a subprocess - using binary mode is safer for reading output
    process = subprocess.Popen(cmd, 
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              cwd=working_dir,
                              env=env)
    
    # Give nginx time to start
    time.sleep(2)
    
    # Check if nginx is running
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        error_msg = f"Error starting nginx: {stderr.decode('utf-8', errors='replace')}"
        log_message(error_msg)
        log_message(f"Nginx stdout: {stdout.decode('utf-8', errors='replace')}")
        log_message(f"Nginx stderr: {stderr.decode('utf-8', errors='replace')}")
        return None
    
    return process

def collect_process_output(process, prefix):
    """Non-blocking collection of process output"""
    if process and process.poll() is None:
        # Get output without blocking
        stdout_data = b""
        stderr_data = b""
        
        # Try to read stdout if available and it's a pipe
        if process.stdout and hasattr(process.stdout, 'fileno'):
            try:
                # Check if there's data to read without blocking
                r, _, _ = select.select([process.stdout], [], [], 0)
                if r:
                    stdout_data = os.read(process.stdout.fileno(), 1024)
            except (ValueError, IOError, OSError):
                pass
        
        # Try to read stderr if available and it's a pipe
        if process.stderr and hasattr(process.stderr, 'fileno'):
            try:
                # Check if there's data to read without blocking
                r, _, _ = select.select([process.stderr], [], [], 0)
                if r:
                    stderr_data = os.read(process.stderr.fileno(), 1024)
            except (ValueError, IOError, OSError):
                pass
            
        # Log any output we got
        if stdout_data:
            log_message(f"{prefix} stdout: {stdout_data.decode('utf-8', errors='replace')}", also_print=False)
        if stderr_data:
            log_message(f"{prefix} stderr: {stderr_data.decode('utf-8', errors='replace')}", also_print=False)

def stop_nginx(process):
    """Stop the nginx process"""
    if process and process.poll() is None:
        log_message("Stopping nginx")
        
        # Collect any remaining output before terminating
        try:
            stdout, stderr = process.communicate(timeout=1)
            if stdout:
                log_message(f"Nginx final stdout: {stdout.decode('utf-8', errors='replace')}", also_print=False)
            if stderr:
                log_message(f"Nginx final stderr: {stderr.decode('utf-8', errors='replace')}", also_print=False)
        except subprocess.TimeoutExpired:
            log_message("Timeout while collecting final output from Nginx")
        
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log_message("Nginx didn't terminate gracefully, killing it")
            process.kill()
        
        # Clean up pid file
        pid_file = PARENT_DIR / "nginx.pid"
        if pid_file.exists():
            pid_file.unlink()
            log_message("Deleted nginx pid file")

def run_wrk_benchmark(url, duration=30, connections=400, threads=12):
    """Run wrk benchmark against the specified URL"""
    cmd = ["wrk", f"-t{threads}", f"-c{connections}", f"-d{duration}s", url]
    cmd_str = ' '.join(cmd)
    log_message(f"Running benchmark: {cmd_str}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        log_message(f"Error running wrk: {result.stderr}")
        return None
    
    # Log the full wrk output
    log_message("\n--- WRK Benchmark Output ---", also_print=False)
    log_message(result.stdout, also_print=False)
    log_message("--- End WRK Benchmark Output ---\n", also_print=False)
    
    return result.stdout

def parse_wrk_output(output):
    """Parse the output from wrk to extract key metrics"""
    if not output:
        return None
    
    metrics = {}
    
    # Extract requests per second
    rps_line = [line for line in output.split('\n') if "Requests/sec:" in line]
    if rps_line:
        metrics['rps'] = float(rps_line[0].split(':')[1].strip())
    
    # Extract latency
    latency_lines = [line for line in output.split('\n') if "Latency" in line]
    if latency_lines:
        parts = latency_lines[0].split()
        metrics['latency_avg'] = parts[1]
        metrics['latency_stdev'] = parts[2]
        metrics['latency_max'] = parts[3]
    
    return metrics

def start_bpftime_controller(prefix):
    """Start the bpftime controller with the specified prefix"""
    controller_path = PARENT_DIR.parent.parent / "build/example/attach_implementation/benchmark/ebpf_controller/nginx_benchmark_ebpf_controller"
    
    if not controller_path.exists():
        log_message(f"Error: bpftime controller not found at {controller_path}")
        return None
    
    cmd = [str(controller_path), prefix]
    cmd_str = ' '.join(cmd)
    log_message(f"Starting bpftime controller with command: {cmd_str}")
    
    # Use binary mode for subprocess to avoid TextIOWrapper issues
    process = subprocess.Popen(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    
    # Give controller time to start
    time.sleep(2)
    
    # Check if controller is running
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        error_msg = f"Error starting bpftime controller: {stderr.decode('utf-8', errors='replace')}"
        log_message(error_msg)
        log_message(f"Controller stdout: {stdout.decode('utf-8', errors='replace')}", also_print=False)
        log_message(f"Controller stderr: {stderr.decode('utf-8', errors='replace')}", also_print=False)
        return None
    
    return process

def start_baseline_controller(prefix):
    """Start the baseline controller with the specified prefix"""
    controller_path = PARENT_DIR.parent.parent / "build/example/attach_implementation/benchmark/baseline_nginx_plugin/nginx_baseline_controller"
    
    if not controller_path.exists():
        log_message(f"Error: Baseline controller not found at {controller_path}")
        log_message("Make sure to build the baseline controller first")
        return None
    
    cmd = [str(controller_path), prefix]
    cmd_str = ' '.join(cmd)
    log_message(f"Starting baseline controller with command: {cmd_str}")
    
    # Use binary mode for subprocess to avoid TextIOWrapper issues
    process = subprocess.Popen(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    
    # Give controller time to start
    time.sleep(2)
    
    # Check if controller is running
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        error_msg = f"Error starting baseline controller: {stderr.decode('utf-8', errors='replace')}"
        log_message(error_msg)
        log_message(f"Controller stdout: {stdout.decode('utf-8', errors='replace')}", also_print=False)
        log_message(f"Controller stderr: {stderr.decode('utf-8', errors='replace')}", also_print=False)
        return None
    
    return process

def stop_controller(process, name):
    """Stop a controller process and capture its output"""
    if process and process.poll() is None:
        log_message(f"Stopping {name} controller")
        
        # Collect any remaining output before terminating
        try:
            stdout, stderr = process.communicate(timeout=1)
            if stdout:
                log_message(f"{name} controller final stdout: {stdout.decode('utf-8', errors='replace')}", also_print=False)
            if stderr:
                log_message(f"{name} controller final stderr: {stderr.decode('utf-8', errors='replace')}", also_print=False)
        except subprocess.TimeoutExpired:
            log_message(f"Timeout while collecting final output from {name} controller")
        
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log_message(f"{name} controller didn't terminate gracefully, killing it")
            process.kill()

def main():
    parser = argparse.ArgumentParser(description="Run nginx benchmarks with different configurations")
    parser.add_argument("--duration", type=int, default=30, help="Duration of each benchmark in seconds")
    parser.add_argument("--connections", type=int, default=4000, help="Number of connections to use")
    parser.add_argument("--threads", type=int, default=12, help="Number of threads to use")
    parser.add_argument("--url-path", type=str, default="/aaaa", help="URL path to test")
    args = parser.parse_args()
    
    # Initialize or clear the log file
    with open(BENCHMARK_LOG, 'w') as f:
        f.write(f"=== Benchmark started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"Duration: {args.duration}s, Connections: {args.connections}, Threads: {args.threads}, URL: {args.url_path}\n\n")
    
    check_prerequisites()
    
    results = {}
    
    # Test with no module
    log_message("\n=== Testing nginx without any module ===")
    nginx_process = start_nginx(NO_MODULE_CONF)
    if nginx_process:
        url = f"http://127.0.0.1:{NO_MODULE_PORT}{args.url_path}"
        output = run_wrk_benchmark(url, args.duration, args.connections, args.threads)
        results['no_module'] = parse_wrk_output(output)
        collect_process_output(nginx_process, "No-module Nginx")
        stop_nginx(nginx_process)
    
    # Test with baseline C module
    log_message("\n=== Testing nginx with baseline C module ===")
    baseline_controller_process = start_baseline_controller(args.url_path)
    if baseline_controller_process:
        nginx_process = start_nginx(BASELINE_CONF)
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
            stop_nginx(nginx_process)
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
    nginx_process = start_nginx(DYNAMIC_LOAD_CONF, env=wasm_env)
    if nginx_process:
        url = f"http://127.0.0.1:{WASM_PORT}{args.url_path}"
        time.sleep(1)
        output = run_wrk_benchmark(url, args.duration, args.connections, args.threads)
        results['wasm'] = parse_wrk_output(output)
        collect_process_output(nginx_process, "WebAssembly Nginx")
        stop_nginx(nginx_process)
    
    # Test with bpftime module
    log_message("\n=== Testing nginx with bpftime module ===")
    bpftime_controller_process = start_bpftime_controller(args.url_path)
    if bpftime_controller_process:
        nginx_process = start_nginx(BPFTIME_CONF)
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
            stop_nginx(nginx_process)
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