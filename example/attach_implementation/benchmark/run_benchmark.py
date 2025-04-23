#!/usr/bin/env python3
"""
Benchmark script for comparing different nginx configurations:
1. With bpftime module (eBPF-based filtering)
2. With baseline C module (direct C implementation)
3. Without any module (baseline performance)

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
from pathlib import Path

# Get the directory of this script
SCRIPT_DIR = Path(__file__).parent.absolute()
# Get the parent directory (attach_implementation)
PARENT_DIR = SCRIPT_DIR.parent

# Configuration
NGINX_BIN = str(PARENT_DIR / "nginx_plugin_output" / "nginx")
BPFTIME_CONF = str(PARENT_DIR / "nginx.conf")
BASELINE_CONF = str(SCRIPT_DIR / "baseline_c_module.conf")
NO_MODULE_CONF = str(SCRIPT_DIR / "no_module.conf")

# Ports
BPFTIME_PORT = 9023
BASELINE_PORT = 9025
NO_MODULE_PORT = 9024

# URLs for testing
TEST_URL_PREFIX = "/aaaa"  # This should match what's allowed in your eBPF program

def check_prerequisites():
    """Check if all required tools are available"""
    required_tools = ["wrk", "nginx"]
    
    for tool in required_tools:
        if shutil.which(tool) is None:
            print(f"Error: {tool} is not installed or not in PATH")
            print(f"Please install {tool} and try again")
            sys.exit(1)
    
    if not os.path.exists(NGINX_BIN):
        print(f"Error: Nginx binary not found at {NGINX_BIN}")
        print("Please build the project first")
        sys.exit(1)

def start_nginx(config_path, working_dir=PARENT_DIR):
    """Start nginx with the specified configuration"""
    cmd = [NGINX_BIN, "-p", str(working_dir), "-c", config_path]
    print(f"Starting nginx with command: {' '.join(cmd)}")
    
    # Start nginx as a subprocess
    process = subprocess.Popen(cmd, 
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              cwd=working_dir)
    
    # Give nginx time to start
    time.sleep(2)
    
    # Check if nginx is running
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        print(f"Error starting nginx: {stderr.decode()}")
        return None
    
    return process

def stop_nginx(process):
    """Stop the nginx process"""
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        
        # Clean up pid file
        pid_file = PARENT_DIR / "nginx.pid"
        if pid_file.exists():
            pid_file.unlink()

def run_wrk_benchmark(url, duration=30, connections=400, threads=12):
    """Run wrk benchmark against the specified URL"""
    cmd = ["wrk", f"-t{threads}", f"-c{connections}", f"-d{duration}s", url]
    print(f"Running benchmark: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running wrk: {result.stderr}")
        return None
    
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

def start_controller(prefix):
    """Start the bpftime controller with the specified prefix"""
    controller_path = PARENT_DIR.parent.parent / "build/example/attach_implementation/benchmark/ebpf_controller/nginx_benchmark_ebpf_controller"
    
    if not controller_path.exists():
        print(f"Error: Controller not found at {controller_path}")
        return None
    
    cmd = [str(controller_path), prefix]
    print(f"Starting controller with command: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    
    # Give controller time to start
    time.sleep(2)
    
    # Check if controller is running
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        print(f"Error starting controller: {stderr.decode()}")
        return None
    
    return process

def main():
    parser = argparse.ArgumentParser(description="Run nginx benchmarks with different configurations")
    parser.add_argument("--duration", type=int, default=30, help="Duration of each benchmark in seconds")
    parser.add_argument("--connections", type=int, default=400, help="Number of connections to use")
    parser.add_argument("--threads", type=int, default=12, help="Number of threads to use")
    parser.add_argument("--url-path", type=str, default="/aaaa", help="URL path to test")
    args = parser.parse_args()
    
    check_prerequisites()
    
    results = {}
    
    # Test with no module
    print("\n=== Testing nginx without any module ===")
    nginx_process = start_nginx(NO_MODULE_CONF)
    if nginx_process:
        url = f"http://127.0.0.1:{NO_MODULE_PORT}{args.url_path}"
        output = run_wrk_benchmark(url, args.duration, args.connections, args.threads)
        results['no_module'] = parse_wrk_output(output)
        stop_nginx(nginx_process)
    
    # Test with baseline C module
    print("\n=== Testing nginx with baseline C module ===")
    nginx_process = start_nginx(BASELINE_CONF)
    if nginx_process:
        url = f"http://127.0.0.1:{BASELINE_PORT}{args.url_path}"
        output = run_wrk_benchmark(url, args.duration, args.connections, args.threads)
        results['baseline'] = parse_wrk_output(output)
        stop_nginx(nginx_process)
    
    # Test with bpftime module
    print("\n=== Testing nginx with bpftime module ===")
    controller_process = start_controller(args.url_path)
    if controller_process:
        nginx_process = start_nginx(BPFTIME_CONF)
        if nginx_process:
            url = f"http://127.0.0.1:{BPFTIME_PORT}{args.url_path}"
            output = run_wrk_benchmark(url, args.duration, args.connections, args.threads)
            results['bpftime'] = parse_wrk_output(output)
            stop_nginx(nginx_process)
        controller_process.terminate()
    
    # Print results
    print("\n=== Benchmark Results ===")
    
    if 'no_module' in results and results['no_module']:
        print(f"\nNginx without module:")
        print(f"  Requests/sec: {results['no_module']['rps']:.2f}")
        print(f"  Latency (avg): {results['no_module']['latency_avg']}")
    
    if 'baseline' in results and results['baseline']:
        print(f"\nNginx with baseline C module:")
        print(f"  Requests/sec: {results['baseline']['rps']:.2f}")
        print(f"  Latency (avg): {results['baseline']['latency_avg']}")
        
        # Calculate overhead compared to no module
        if 'no_module' in results and results['no_module']:
            overhead = (1 - results['baseline']['rps'] / results['no_module']['rps']) * 100
            print(f"  Overhead vs no module: {overhead:.2f}%")
    
    if 'bpftime' in results and results['bpftime']:
        print(f"\nNginx with bpftime module:")
        print(f"  Requests/sec: {results['bpftime']['rps']:.2f}")
        print(f"  Latency (avg): {results['bpftime']['latency_avg']}")
        
        # Calculate overhead compared to no module
        if 'no_module' in results and results['no_module']:
            overhead = (1 - results['bpftime']['rps'] / results['no_module']['rps']) * 100
            print(f"  Overhead vs no module: {overhead:.2f}%")
        
        # Calculate overhead compared to baseline
        if 'baseline' in results and results['baseline']:
            overhead = (1 - results['bpftime']['rps'] / results['baseline']['rps']) * 100
            print(f"  Overhead vs baseline C module: {overhead:.2f}%")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(0) 