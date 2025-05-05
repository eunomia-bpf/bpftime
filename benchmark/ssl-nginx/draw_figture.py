#!/usr/bin/env python3
import subprocess
import os
import json
import time
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Configuration
SIZES = {
    "16b": 16,
    "1kb": 1024,
    "2kb": 2 * 1024,
    "4kb": 4 * 1024,
    "16kb": 16 * 1024,
    "128kb": 128 * 1024,
    "256kb": 256 * 1024
}

BENCHMARK_CMD = ["python", "benchmark/ssl-nginx/benchmark.py"]
OUTPUT_DIR = "benchmark/ssl-nginx"
INDEX_HTML_PATH = "benchmark/ssl-nginx/index.html"

def create_dir_if_not_exists(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

def generate_test_file(size_name, size_bytes):
    """Generate a single HTML test file with the specified size"""
    print(f"Generating test file of size {size_name}...")
    
    # Create HTML content
    html_start = "<html>"
    html_end = "</html>"
    
    # Calculate content size
    content_size = size_bytes - len(html_start) - len(html_end)
    if content_size < 0:
        content_size = 0
    
    # Generate content
    content = "X" * content_size
    
    # Write to file
    with open(INDEX_HTML_PATH, "w") as f:
        f.write(html_start + content + html_end)
    
    actual_size = os.path.getsize(INDEX_HTML_PATH)
    print(f"  Created {INDEX_HTML_PATH} ({actual_size} bytes)")

def run_benchmark(size_name, size_bytes):
    """Run the benchmark for a specific file size"""
    print(f"\n=== Running benchmark for {size_name} file ===")
    
    # Generate the test file
    generate_test_file(size_name, size_bytes)
    
    # Run the benchmark
    result = subprocess.run(BENCHMARK_CMD, capture_output=True, text=True)
    
    # Extract results
    output = result.stdout
    print(output)
    
    # Parse results
    results = {}
    for test_type in ["baseline", "kernel_sslsniff", "bpftime_sslsniff"]:
        match = re.search(rf"{test_type.replace('_', ' ').title()}:\s+Requests/sec \(mean\):\s+(\d+\.\d+)", output, re.IGNORECASE | re.MULTILINE)
        if match:
            results[test_type] = float(match.group(1))
        else:
            results[test_type] = None
    
    return {
        "size_name": size_name,
        "size_bytes": size_bytes,
        "results": results,
        "raw_output": output
    }

def save_results(all_results):
    """Save results to JSON and TXT files"""
    create_dir_if_not_exists(OUTPUT_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    json_path = f"{OUTPUT_DIR}/size_benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Save as TXT
    txt_path = f"{OUTPUT_DIR}/size_benchmark_{timestamp}.txt"
    with open(txt_path, "w") as f:
        f.write(f"File Size Benchmark Results - {timestamp}\n\n")
        
        # Table header
        f.write(f"{'Size':<10} {'Baseline':<15} {'Kernel':<15} {'BPFtime':<15} {'Kernel Impact':<15} {'BPFtime Impact':<15}\n")
        f.write("-" * 85 + "\n")
        
        # Table data
        for result in all_results:
            size = result["size_name"]
            baseline = result["results"]["baseline"]
            kernel = result["results"]["kernel_sslsniff"]
            bpftime = result["results"]["bpftime_sslsniff"]
            
            # Calculate impacts
            kernel_impact = ((baseline - kernel) / baseline * 100) if baseline and kernel else "N/A"
            bpftime_impact = ((baseline - bpftime) / baseline * 100) if baseline and bpftime else "N/A"
            
            # Format numbers
            baseline_str = f"{baseline:.2f}" if baseline else "N/A"
            kernel_str = f"{kernel:.2f}" if kernel else "N/A"
            bpftime_str = f"{bpftime:.2f}" if bpftime else "N/A"
            kernel_impact_str = f"{kernel_impact:.2f}%" if isinstance(kernel_impact, float) else kernel_impact
            bpftime_impact_str = f"{bpftime_impact:.2f}%" if isinstance(bpftime_impact, float) else bpftime_impact
            
            f.write(f"{size:<10} {baseline_str:<15} {kernel_str:<15} {bpftime_str:<15} {kernel_impact_str:<15} {bpftime_impact_str:<15}\n")
        
        # Add raw output
        f.write("\n\nRaw output for each test:\n")
        for result in all_results:
            f.write(f"\n\n=== {result['size_name']} ===\n")
            f.write(result["raw_output"])
    
    print(f"\nResults saved to {json_path} and {txt_path}")
    return json_path, txt_path

def plot_results(all_results, output_dir):
    """Create plots based on the benchmark results"""
    sizes = [result["size_name"] for result in all_results]
    baseline_values = [result["results"]["baseline"] if result["results"]["baseline"] else 0 for result in all_results]
    kernel_values = [result["results"]["kernel_sslsniff"] if result["results"]["kernel_sslsniff"] else 0 for result in all_results]
    bpftime_values = [result["results"]["bpftime_sslsniff"] if result["results"]["bpftime_sslsniff"] else 0 for result in all_results]
    
    # Absolute performance plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(sizes))
    width = 0.25
    
    plt.bar(x - width, baseline_values, width, label='Baseline')
    plt.bar(x, kernel_values, width, label='Kernel sslsniff')
    plt.bar(x + width, bpftime_values, width, label='BPFtime sslsniff')
    
    plt.xlabel('File Size')
    plt.ylabel('Requests/sec')
    plt.title('Performance by File Size')
    plt.xticks(x, sizes)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save absolute performance plot
    abs_plot_path = f"{output_dir}/absolute_performance.png"
    plt.savefig(abs_plot_path)

def main():
    try:
        # Make sure we have the output directory
        create_dir_if_not_exists(OUTPUT_DIR)
        
        # Run benchmarks for each file size
        all_results = []
        for size_name, size_bytes in SIZES.items():
            result = run_benchmark(size_name, size_bytes)
            all_results.append(result)
            time.sleep(2)  # Brief pause between benchmarks
        
        # Save results
        json_path, txt_path = save_results(all_results)
        
        # Create plots
        plot_results(all_results, OUTPUT_DIR)
        
        print("\nBenchmark completed successfully!")
        
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 