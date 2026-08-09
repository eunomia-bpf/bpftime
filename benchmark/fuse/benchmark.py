#!/usr/bin/env python3
import subprocess
import time
import os
import statistics
import textwrap
from tabulate import tabulate
from pathlib import Path

# Configuration
DATA_DIR = "/root/bpftime-evaluation/fuse/data"
NUM_RUNS = 10  # Number of times to run each test
TESTS = [
    {
        "name": "Passthrough, fstat",
        "command": "stat -f {dir}",
        "description": "Passthrough filesystem with fstat operation"
    },
    {
        "name": "LoggedFS, fstat",
        "command": "stat -f {dir}",
        "description": "LoggedFS filesystem with fstat operation" 
    },
    {
        "name": "LoggedFS, openat",
        "command": "find {dir} -type f -exec cat {{}} \\; > /dev/null 2>&1",
        "description": "LoggedFS filesystem with openat operation"
    },
    {
        "name": "Passthrough, find",
        "command": "find {dir} 2>/dev/null 1>/dev/null",
        "description": "Passthrough filesystem with find operation"
    }
]

def run_native_test(test, data_dir):
    """Run a test without bpftime and measure the execution time."""
    cmd = test["command"].format(dir=data_dir)
    times = []
    print(f"Running native test: {test['name']}")
    
    for i in range(NUM_RUNS):
        print(f"  Run {i+1}/{NUM_RUNS}...", end="", flush=True)
        start_time = time.time()
        subprocess.run(cmd, shell=True)
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        print(f" {elapsed:.3f}s")
    
    return times

def run_bpftime_test(test, data_dir):
    """Run a test with bpftime and measure the execution time."""
    cmd = test["command"].format(dir=data_dir)
    bpftime_cmd = (
        f"AGENT_SO=/root/zys/bpftime/build/runtime/agent/libbpftime-agent.so "
        f"LD_PRELOAD=/root/zys/bpftime/build/runtime/agent-transformer/libbpftime-agent-transformer.so "
        f"{cmd}"
    )
    times = []
    print(f"Running bpftime test: {test['name']}")
    
    # Start the syscall server for bpftime
    server_process = subprocess.Popen(
        "LD_PRELOAD=/root/zys/bpftime/build/runtime/syscall-server/libbpftime-syscall-server.so "
        "./fs-filter-cache/fs-cache",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Allow the server to start
    time.sleep(1)
    
    try:
        for i in range(NUM_RUNS):
            print(f"  Run {i+1}/{NUM_RUNS}...", end="", flush=True)
            start_time = time.time()
            subprocess.run(bpftime_cmd, shell=True)
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            print(f" {elapsed:.3f}s")
    finally:
        # Terminate the server
        server_process.terminate()
        server_process.wait(timeout=5)
    
    return times

def calculate_statistics(times):
    """Calculate statistics for the measured times."""
    if not times:
        return {"avg": 0, "median": 0, "min": 0, "max": 0, "std_dev": 0}
    
    return {
        "avg": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0
    }

def run_benchmark():
    """Run the benchmark for all tests."""
    results = []
    
    # Make sure the fs-filter-cache is built
    if not os.path.exists("./fs-filter-cache/fs-cache"):
        print("Building fs-filter-cache...")
        os.makedirs("fs-filter-cache", exist_ok=True)
        subprocess.run("cd bpf && make", shell=True, check=True)
        # Add appropriate commands to build fs-filter-cache
    
    for test in TESTS:
        print(f"\n=== Running test: {test['name']} ===")
        print(f"Description: {test['description']}")
        
        # Run native test
        native_times = run_native_test(test, DATA_DIR)
        native_stats = calculate_statistics(native_times)
        
        # Run bpftime test
        bpftime_times = run_bpftime_test(test, DATA_DIR)
        bpftime_stats = calculate_statistics(bpftime_times)
        
        results.append({
            "name": test["name"],
            "native_avg": native_stats["avg"],
            "bpftime_avg": bpftime_stats["avg"],
            "speedup": native_stats["avg"] / bpftime_stats["avg"] if bpftime_stats["avg"] > 0 else 0
        })
        
        print(f"\nResults for {test['name']}:")
        print(f"  Native:  avg={native_stats['avg']:.3f}s, median={native_stats['median']:.3f}s, "
              f"min={native_stats['min']:.3f}s, max={native_stats['max']:.3f}s, "
              f"std_dev={native_stats['std_dev']:.3f}s")
        print(f"  BPFtime: avg={bpftime_stats['avg']:.3f}s, median={bpftime_stats['median']:.3f}s, "
              f"min={bpftime_stats['min']:.3f}s, max={bpftime_stats['max']:.3f}s, "
              f"std_dev={bpftime_stats['std_dev']:.3f}s")
        print(f"  Speedup: {native_stats['avg'] / bpftime_stats['avg']:.2f}x")
    
    return results

def print_results_table(results):
    """Print a table of results."""
    table_data = []
    
    for result in results:
        table_data.append([
            result["name"],
            f"{result['native_avg']:.2f}",
            f"{result['bpftime_avg']:.3f}"
        ])
    
    print("\n=== FUSE operation latency ===")
    print(tabulate(table_data, headers=["Test", "Native (s)", "bpftime (s)"], tablefmt="grid"))
    
    # Also print in the format of the table shown in the requirements
    print("\nTable 2: FUSE operation latency.")
    for row in table_data:
        print(f"{row[0]} {row[1]} {row[2]}")

if __name__ == "__main__":
    print("Starting FUSE benchmark...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Number of runs per test: {NUM_RUNS}")
    
    try:
        # Check if tabulate is installed
        import tabulate
    except ImportError:
        print("Installing tabulate package...")
        subprocess.run("pip install tabulate", shell=True, check=True)
        from tabulate import tabulate
    
    # Run the benchmark
    results = run_benchmark()
    
    # Print the results
    print_results_table(results) 