#!/usr/bin/env python3
"""
CUDA Benchmark Runner for bpftime
Runs benchmarks using JSON configuration.
"""

import subprocess
import re
import sys
import os
import time
import json
from datetime import datetime
from typing import Optional, Dict, List, Any

# Constants
LOG_FILE = "micro_bench.log"
RESULT_FILE = "micro_result.json"
DEFAULT_CONFIG = "bench_config.json"

def log(msg: str, log_file=None):
    """Log message to both stdout and file."""
    print(msg)
    if log_file:
        log_file.write(msg + '\n')
        log_file.flush()

def parse_benchmark_output(output: str) -> Optional[float]:
    """Parse the average kernel time from benchmark output."""
    match = re.search(r'Average kernel time:\s+([\d.]+)\s+us', output)
    if match:
        return float(match.group(1))
    return None

def get_gpu_name() -> str:
    """Get the GPU device name."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0]
    except:
        pass
    return "Unknown GPU"

def run_baseline(bench_dir: str, vec_add_args: str, log_file=None) -> Optional[float]:
    """Run baseline benchmark without instrumentation."""
    log(f"Running baseline benchmark with args: {vec_add_args}", log_file)
    try:
        cmd = [f'{bench_dir}/vec_add'] + (vec_add_args.split() if vec_add_args else [])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=bench_dir
        )
        log(f"STDOUT:\n{result.stdout}", log_file)
        log(f"STDERR:\n{result.stderr}", log_file)

        if result.returncode != 0:
            log(f"Baseline failed with return code {result.returncode}", log_file)
            return None

        avg_time = parse_benchmark_output(result.stdout)
        if avg_time:
            log(f"✓ Baseline: {avg_time:.2f} μs", log_file)
        return avg_time
    except Exception as e:
        log(f"Error running baseline: {e}", log_file)
        return None

def run_bpf_test(bench_dir: str, build_dir: str, cuda_probe_args: str, vec_add_args: str, log_file=None) -> Optional[float]:
    """Run benchmark with BPF instrumentation."""
    log(f"Running BPF test: probe_args={cuda_probe_args}, vec_add_args={vec_add_args}", log_file)

    probe_process = None
    try:
        # Start BPF probe in background
        syscall_server = f"{build_dir}/runtime/syscall-server/libbpftime-syscall-server.so"
        agent = f"{build_dir}/runtime/agent/libbpftime-agent.so"

        if not os.path.exists(syscall_server):
            log(f"Error: {syscall_server} not found", log_file)
            return None
        if not os.path.exists(agent):
            log(f"Error: {agent} not found", log_file)
            return None

        probe_env = os.environ.copy()
        probe_env['BPFTIME_LOG_OUTPUT'] = 'console'
        probe_env['LD_PRELOAD'] = syscall_server

        probe_cmd = [f'{bench_dir}/cuda_probe'] + cuda_probe_args.split()
        probe_log = f"{bench_dir}/.cuda_probe.stderr"
        with open(probe_log, 'w') as probe_stderr:
            probe_process = subprocess.Popen(
                probe_cmd,
                env=probe_env,
                stdout=probe_stderr,
                stderr=probe_stderr,
                cwd=bench_dir
            )

        # Wait for probe to initialize
        time.sleep(2)

        # Log probe startup
        if os.path.exists(probe_log):
            with open(probe_log, 'r') as f:
                probe_output = f.read()
                log(f"Probe startup log:\n{probe_output}", log_file)

        # Run benchmark with BPF agent
        agent_env = os.environ.copy()
        agent_env['BPFTIME_LOG_OUTPUT'] = 'console'
        agent_env['LD_PRELOAD'] = agent

        vec_add_cmd = [f'{bench_dir}/vec_add'] + (vec_add_args.split() if vec_add_args else [])
        result = subprocess.run(
            vec_add_cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=bench_dir,
            env=agent_env
        )

        log(f"STDOUT:\n{result.stdout}", log_file)
        log(f"STDERR:\n{result.stderr}", log_file)

        if result.returncode != 0:
            log(f"BPF benchmark failed with return code {result.returncode}", log_file)
            return None

        avg_time = parse_benchmark_output(result.stdout)
        if avg_time:
            log(f"✓ BPF test: {avg_time:.2f} μs", log_file)
        return avg_time

    except Exception as e:
        log(f"Error running BPF benchmark: {e}", log_file)
        return None
    finally:
        if probe_process:
            try:
                probe_process.terminate()
                probe_process.wait(timeout=5)
            except:
                probe_process.kill()

        # Clean up probe log
        if os.path.exists(probe_log):
            try:
                os.remove(probe_log)
            except:
                pass

def load_config(config_path: str) -> Dict[str, Any]:
    """Load benchmark configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_results(results: Dict[str, Any], output_path: str):
    """Save benchmark results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def print_results_table(results: Dict[str, Any], log_file=None):
    """Print the results in a formatted table."""
    log("\n" + "="*80, log_file)
    log("CUDA Benchmark Results", log_file)
    log("="*80 + "\n", log_file)

    log(f"Device: {results.get('device', 'Unknown')}", log_file)
    log(f"Timestamp: {results.get('timestamp', 'Unknown')}\n", log_file)

    baseline_time = results.get('baseline_time')

    log(f"{'Test Name':<40} {'Avg Time (μs)':<15} {'Overhead':<15}", log_file)
    log("-"*70, log_file)

    # Baseline
    if baseline_time:
        log(f"{'Baseline (no probe)':<40} {baseline_time:<15.2f} {'-':<15}", log_file)
    else:
        log(f"{'Baseline (no probe)':<40} {'FAILED':<15} {'-':<15}", log_file)

    # Test cases
    for test in results.get('tests', []):
        name = test['name']
        avg_time = test.get('avg_time_us')

        if avg_time and baseline_time:
            overhead = f"{avg_time/baseline_time:.2f}x (+{((avg_time/baseline_time-1)*100):.1f}%)"
            log(f"{name:<40} {avg_time:<15.2f} {overhead:<15}", log_file)
        elif avg_time:
            log(f"{name:<40} {avg_time:<15.2f} {'-':<15}", log_file)
        else:
            log(f"{name:<40} {'FAILED':<15} {'-':<15}", log_file)

    log("\n" + "="*80 + "\n", log_file)

def main():
    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    bench_dir = script_dir
    build_dir = os.path.join(repo_root, 'build')
    log_path = os.path.join(bench_dir, LOG_FILE)
    result_path = os.path.join(bench_dir, RESULT_FILE)

    # Parse arguments
    config_file = DEFAULT_CONFIG
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    config_path = os.path.join(bench_dir, config_file)
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        print(f"Usage: {sys.argv[0]} [config.json]")
        sys.exit(1)

    # Load configuration
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Extract config
    vec_add_args = config.get('vec_add_args', '1024 10')
    test_cases = config.get('test_cases', [])

    with open(log_path, 'w') as log_file:
        log("="*80, log_file)
        log("CUDA Benchmark Suite", log_file)
        log("="*80, log_file)
        log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
        log(f"Config file: {config_file}", log_file)
        log(f"Vec_add args: {vec_add_args}", log_file)
        log("="*80 + "\n", log_file)

        # Get GPU name
        device = get_gpu_name()
        log(f"GPU: {device}\n", log_file)

        # Prepare results
        results = {
            'device': device,
            'timestamp': datetime.now().isoformat(),
            'config_file': config_file,
            'vec_add_args': vec_add_args,
            'baseline_time': None,
            'tests': []
        }

        # Run baseline
        results['baseline_time'] = run_baseline(bench_dir, vec_add_args, log_file)
        log("", log_file)

        # Run test cases
        for test_case in test_cases:
            name = test_case.get('name', 'Unknown')
            cuda_probe_args = test_case.get('cuda_probe_args', '')
            test_vec_add_args = test_case.get('vec_add_args', vec_add_args)

            log(f"Running test: {name}", log_file)
            avg_time = run_bpf_test(bench_dir, build_dir, cuda_probe_args, test_vec_add_args, log_file)

            results['tests'].append({
                'name': name,
                'cuda_probe_args': cuda_probe_args,
                'vec_add_args': test_vec_add_args,
                'avg_time_us': avg_time
            })
            log("", log_file)

        # Save results
        save_results(results, result_path)
        log(f"Results saved to: {RESULT_FILE}", log_file)

        # Print results table
        print_results_table(results, log_file)

        log(f"Log saved to: {LOG_FILE}", log_file)

if __name__ == '__main__':
    main()
