#!/usr/bin/env python3
"""
Redis Durability Tuning Benchmark

This script benchmarks different Redis durability approaches:
1. Standard Redis with no AOF, everysec, and alwayson configs
2. Custom durability approaches using BPFtime:
   - io_uring batched I/O
   - delayed-fsync
   - fsync-fast-notify

The script measures throughput (operations/sec) and latency.
"""

import os
import time
import subprocess
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REDIS_DIR = Path("redis")
REDIS_SERVER = REDIS_DIR / "src" / "redis-server"
REDIS_CLI = REDIS_DIR / "src" / "redis-cli"
REDIS_BENCHMARK = REDIS_DIR / "src" / "redis-benchmark"

# Configuration for different durability modes
CONFIGS = {
    "standard": {
        "no-aof": {
            "config_mods": ["appendfsync no"], 
            "description": "Standard Redis with no AOF"
        },
        "everysec": {
            "config_mods": ["appendfsync everysec"], 
            "description": "Standard Redis with fsync every second"
        },
        "alwayson": {
            "config_mods": ["appendfsync always"], 
            "description": "Standard Redis with fsync for every write"
        },
    },
    "custom": {
        "iouring-batch-10": {
            "dir": "poc-iouring-minimal",
            "env": {"BATCH_SIZE": "10"},
            "description": "io_uring with batch size 10"
        },
        "iouring-batch-100": {
            "dir": "poc-iouring-minimal",
            "env": {"BATCH_SIZE": "100"},
            "description": "io_uring with batch size 100"
        },
        "iouring-batch-1000": {
            "dir": "poc-iouring-minimal",
            "env": {"BATCH_SIZE": "1000"},
            "description": "io_uring with batch size 1000"
        },
        "delayed-fsync": {
            "dir": "delayed-fsync",
            "env": {},
            "description": "Delayed fsync approach"
        },
        "fsync-fast-notify": {
            "dir": "fsync-fast-notify",
            "env": {},
            "description": "Fast-path optimization for fsync"
        },
    }
}

class RedisBenchmark:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.results = {}
        
    def update_config(self, config_mods):
        """Update Redis config file with specific modifications"""
        config_path = self.base_dir / "redis.conf"
        temp_config_path = self.base_dir / "redis.conf.tmp"
        
        with open(config_path, 'r') as f:
            config_content = f.readlines()
        
        # Comment out all appendfsync lines
        for i, line in enumerate(config_content):
            if line.strip().startswith("appendfsync "):
                config_content[i] = "# " + line
        
        # Add our specific config modifications
        with open(temp_config_path, 'w') as f:
            f.writelines(config_content)
            for mod in config_mods:
                f.write(f"{mod}\n")
        
        os.rename(temp_config_path, config_path)
    
    def start_redis(self, config_name, config):
        """Start Redis server with the specified configuration"""
        print(f"Starting Redis with configuration: {config_name}")
        
        if config_name.startswith("standard:"):
            # Standard Redis configuration
            config_details = CONFIGS["standard"][config_name.split(":")[1]]
            self.update_config(config_details["config_mods"])
            
            cmd = [str(REDIS_SERVER), str(self.base_dir / "redis.conf")]
            self.redis_process = subprocess.Popen(cmd)
            time.sleep(2)  # Give Redis time to start
            
        else:
            # Custom BPFtime implementation
            config_details = CONFIGS["custom"][config_name]
            self.update_config(["appendfsync always"])  # Always use alwayson for custom implementations
            
            # Build the custom implementation if needed
            impl_dir = self.base_dir / config_details["dir"]
            if (impl_dir / "Makefile").exists():
                subprocess.run(["make", "-C", str(impl_dir)], check=True)
            
            # Create environment variables for the process
            env = os.environ.copy()
            env.update(config_details["env"])
            
            # Start Redis with the BPFtime extension
            cmd = [
                "AGENT_SO=build/runtime/agent/libbpftime-agent.so",
                "LD_PRELOAD=build/runtime/agent-transformer/libbpftime-agent-transformer.so",
                str(REDIS_SERVER),
                str(self.base_dir / "redis.conf")
            ]
            self.redis_process = subprocess.Popen(" ".join(cmd), shell=True, env=env)
            time.sleep(2)  # Give Redis time to start
    
    def stop_redis(self):
        """Stop the Redis server"""
        if hasattr(self, 'redis_process'):
            self.redis_process.terminate()
            self.redis_process.wait()
            print("Redis server stopped")
    
    def run_benchmark(self, clients=50, requests=100000, test="set", data_size=100):
        """Run Redis benchmark with specified parameters"""
        cmd = [
            str(REDIS_BENCHMARK),
            "-c", str(clients),
            "-n", str(requests),
            "-t", test,
            "-d", str(data_size),
            "--csv"
        ]
        
        print(f"Running benchmark: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse the CSV output
        lines = result.stdout.strip().split('\n')
        data = {}
        for line in lines[1:]:  # Skip header
            parts = line.split(',')
            if len(parts) >= 5:
                test_name = parts[0].strip('"')
                ops_per_sec = float(parts[1].strip('"'))
                data[test_name] = {
                    "ops_per_sec": ops_per_sec,
                    "avg_latency_ms": float(parts[2].strip('"')),
                    "min_latency_ms": float(parts[3].strip('"')),
                    "max_latency_ms": float(parts[4].strip('"'))
                }
        
        return data
    
    def benchmark_all(self, clients=50, requests=100000, test="set", data_size=100):
        """Benchmark all configurations"""
        self.results = {}
        
        # Standard Redis configurations
        for config_name, config in CONFIGS["standard"].items():
            full_config_name = f"standard:{config_name}"
            try:
                self.start_redis(full_config_name, config)
                self.results[full_config_name] = {
                    "description": config["description"],
                    "benchmark": self.run_benchmark(clients, requests, test, data_size)
                }
            finally:
                self.stop_redis()
        
        # Custom durability configurations
        for config_name, config in CONFIGS["custom"].items():
            try:
                self.start_redis(config_name, config)
                self.results[config_name] = {
                    "description": config["description"],
                    "benchmark": self.run_benchmark(clients, requests, test, data_size)
                }
            finally:
                self.stop_redis()
        
        return self.results
    
    def save_results(self, output_file="benchmark_results.json"):
        """Save benchmark results to a JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    def plot_results(self, output_file="benchmark_results.png"):
        """Plot benchmark results"""
        if not self.results:
            print("No results to plot")
            return
        
        labels = []
        throughputs = []
        latencies = []
        colors = []
        
        # Standard configurations in blue
        for config_name, result in sorted(self.results.items()):
            if config_name.startswith("standard:"):
                labels.append(result["description"])
                test_result = next(iter(result["benchmark"].values()))
                throughputs.append(test_result["ops_per_sec"])
                latencies.append(test_result["avg_latency_ms"])
                colors.append('blue')
        
        # Custom configurations in green
        for config_name, result in sorted(self.results.items()):
            if not config_name.startswith("standard:"):
                labels.append(result["description"])
                test_result = next(iter(result["benchmark"].values()))
                throughputs.append(test_result["ops_per_sec"])
                latencies.append(test_result["avg_latency_ms"])
                colors.append('green')
        
        # Create figure with throughput and latency plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot throughput
        x = np.arange(len(labels))
        ax1.bar(x, throughputs, color=colors)
        ax1.set_ylabel('Operations per second')
        ax1.set_title('Redis Durability Performance: Throughput')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        
        # Plot latency
        ax2.bar(x, latencies, color=colors)
        ax2.set_ylabel('Average Latency (ms)')
        ax2.set_title('Redis Durability Performance: Latency')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Standard Redis'),
            Patch(facecolor='green', label='Custom with BPFtime')
        ]
        fig.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Redis Durability Benchmark')
    parser.add_argument('--clients', type=int, default=50, help='Number of parallel clients')
    parser.add_argument('--requests', type=int, default=100000, help='Number of requests')
    parser.add_argument('--test', default='set', help='Test type (set, get, etc.)')
    parser.add_argument('--data-size', type=int, default=100, help='Data size in bytes')
    parser.add_argument('--output', default='benchmark_results.json', help='Output JSON file')
    parser.add_argument('--plot', default='benchmark_results.png', help='Output plot file')
    
    args = parser.parse_args()
    
    benchmark = RedisBenchmark(os.path.dirname(os.path.abspath(__file__)))
    benchmark.benchmark_all(args.clients, args.requests, args.test, args.data_size)
    benchmark.save_results(args.output)
    benchmark.plot_results(args.plot)

if __name__ == "__main__":
    main() 