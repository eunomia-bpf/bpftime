#!/usr/bin/env python3
"""
CUDA Benchmark Runner for bpftime
Runs baseline, BPF, and NVBit benchmarks and prints performance comparison table.
"""

import subprocess
import re
import sys
import os
import time
import signal
from datetime import datetime
from typing import Optional, Dict

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

def run_baseline(bench_dir: str, iterations: int = 3, log_file=None) -> Optional[float]:
    """Run baseline benchmark without instrumentation."""
    log("Running baseline benchmark...", log_file)
    try:
        result = subprocess.run(
            [f'{bench_dir}/vec_add', str(iterations)],
            capture_output=True,
            text=True,
            timeout=60,
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

def run_bpf(bench_dir: str, build_dir: str, mode: str = "empty", iterations: int = 3, log_file=None) -> Optional[float]:
    """Run benchmark with BPF instrumentation."""
    log(f"Running BPF benchmark (mode: {mode})...", log_file)

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

        probe_log = f"{bench_dir}/cuda_probe_{mode}.stderr"
        with open(probe_log, 'w') as probe_stderr:
            probe_process = subprocess.Popen(
                [f'{bench_dir}/cuda_probe', mode],
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
        agent_env['BPFTIME_LOG_OUTPUT'] = 'console'  # Note: agent uses BPFTIME_LOG_OUTPUT not BPFTIME_LOG_OUTPUT
        agent_env['LD_PRELOAD'] = agent

        result = subprocess.run(
            [f'{bench_dir}/vec_add', str(iterations)],
            capture_output=True,
            text=True,
            timeout=60,
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
            log(f"✓ BPF ({mode}): {avg_time:.2f} μs", log_file)
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
        probe_log = f"{bench_dir}/cuda_probe_{mode}.stderr"
        if os.path.exists(probe_log):
            try:
                os.remove(probe_log)
            except:
                pass

def run_nvbit(bench_dir: str, iterations: int = 3, log_file=None) -> Optional[float]:
    """Run benchmark with NVBit instrumentation."""
    log("Running NVBit benchmark...", log_file)

    nvbit_so = f"{bench_dir}/nvbit_vec_add.so"
    if not os.path.exists(nvbit_so):
        log(f"NVBit not found at {nvbit_so}, skipping", log_file)
        return None

    try:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        env['LD_PRELOAD'] = './nvbit_vec_add.so'

        result = subprocess.run(
            [f'{bench_dir}/vec_add', str(iterations)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=bench_dir,
            env=env
        )

        log(f"STDOUT:\n{result.stdout}", log_file)
        log(f"STDERR:\n{result.stderr}", log_file)

        if result.returncode != 0:
            log(f"NVBit benchmark failed with return code {result.returncode}", log_file)
            return None

        avg_time = parse_benchmark_output(result.stdout)
        if avg_time:
            log(f"✓ NVBit: {avg_time:.2f} μs", log_file)
        return avg_time

    except Exception as e:
        log(f"Error running NVBit benchmark: {e}", log_file)
        return None

def print_results_table(device: str, results: Dict[str, Optional[float]], comprehensive: bool = False, log_file=None):
    """Print the results in a formatted table."""
    log("\n" + "="*80, log_file)
    log("CUDA Benchmark Results", log_file)
    log("="*80 + "\n", log_file)

    baseline = results.get('baseline')
    log(f"Device: {device}\n", log_file)

    if comprehensive:
        # Comprehensive benchmark results
        log(f"{'Test Mode':<30} {'Avg Time (μs)':<15} {'Overhead':<15}", log_file)
        log("-"*60, log_file)

        # Baseline
        if baseline:
            log(f"{'Baseline (no probe)':<30} {baseline:<15.2f} {'-':<15}", log_file)
        else:
            log(f"{'Baseline (no probe)':<30} {'FAILED':<15} {'-':<15}", log_file)

        # All BPF test modes
        bpf_modes = [
            ('bpf_empty', 'Empty probe'),
            ('bpf_entry', 'Entry only'),
            ('bpf_exit', 'Exit only'),
            ('bpf_both', 'Entry + Exit'),
            ('bpf_ringbuf', 'Ring buffer'),
            ('bpf_array_update', 'Array map update'),
            ('bpf_array_lookup', 'Array map lookup'),
            ('bpf_hash_update', 'Hash map update'),
            ('bpf_hash_lookup', 'Hash map lookup'),
            ('bpf_hash_delete', 'Hash map delete'),
            ('bpf_percpu_array_update', 'PerCPU array update'),
            ('bpf_percpu_array_lookup', 'PerCPU array lookup'),
            ('bpf_percpu_hash_update', 'PerCPU hash update'),
            ('bpf_percpu_hash_lookup', 'PerCPU hash lookup'),
        ]

        for key, name in bpf_modes:
            val = results.get(key)
            if val and baseline:
                overhead = f"{val/baseline:.2f}x (+{((val/baseline-1)*100):.1f}%)"
                log(f"{name:<30} {val:<15.2f} {overhead:<15}", log_file)
            elif val:
                log(f"{name:<30} {val:<15.2f} {'-':<15}", log_file)
            else:
                log(f"{name:<30} {'FAILED':<15} {'-':<15}", log_file)

        # NVBit
        nvbit = results.get('nvbit')
        if nvbit and baseline:
            overhead = f"{nvbit/baseline:.2f}x (+{((nvbit/baseline-1)*100):.1f}%)"
            log(f"{'NVBit':<30} {nvbit:<15.2f} {overhead:<15}", log_file)
        elif nvbit:
            log(f"{'NVBit':<30} {nvbit:<15.2f} {'-':<15}", log_file)

    else:
        # Simple benchmark results
        bpf = results.get('bpf')
        nvbit = results.get('nvbit')

        log(f"{'Method':<20} {'Avg Time (μs)':<20} {'Overhead':<15}", log_file)
        log("-"*55, log_file)

        # Baseline
        if baseline:
            log(f"{'Baseline':<20} {baseline:<20.2f} {'-':<15}", log_file)
        else:
            log(f"{'Baseline':<20} {'FAILED':<20} {'-':<15}", log_file)

        # BPF
        if bpf and baseline:
            overhead = f"{bpf/baseline:.2f}x"
            log(f"{'BPF (bpftime)':<20} {bpf:<20.2f} {overhead:<15}", log_file)
        elif bpf:
            log(f"{'BPF (bpftime)':<20} {bpf:<20.2f} {'-':<15}", log_file)
        else:
            log(f"{'BPF (bpftime)':<20} {'FAILED':<20} {'-':<15}", log_file)

        # NVBit
        if nvbit and baseline:
            overhead = f"{nvbit/baseline:.2f}x"
            log(f"{'NVBit':<20} {nvbit:<20.2f} {overhead:<15}", log_file)
        elif nvbit:
            log(f"{'NVBit':<20} {nvbit:<20.2f} {'-':<15}", log_file)
        else:
            log(f"{'NVBit':<20} {'FAILED':<20} {'-':<15}", log_file)

    log("\n" + "="*80 + "\n", log_file)

def main():
    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    bench_dir = script_dir
    build_dir = os.path.join(repo_root, 'build')

    if not os.path.exists(bench_dir):
        print("Error: benchmark/gpu directory not found")
        sys.exit(1)

    # Parse arguments
    iterations = 3
    comprehensive = False

    for arg in sys.argv[1:]:
        if arg == '--comprehensive' or arg == '-c':
            comprehensive = True
        else:
            try:
                iterations = int(arg)
            except ValueError:
                print(f"Invalid argument: {arg}")
                print(f"Usage: {sys.argv[0]} [iterations] [--comprehensive|-c]")
                sys.exit(1)

    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"benchmark_{timestamp}.log"
    log_path = os.path.join(bench_dir, log_filename)

    with open(log_path, 'w') as log_file:
        log("="*80, log_file)
        log("CUDA Benchmark Suite", log_file)
        log("="*80, log_file)
        log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
        log(f"Iterations: {iterations}", log_file)
        if comprehensive:
            log("Mode: Comprehensive (all test scenarios)", log_file)
        else:
            log("Mode: Quick (baseline + empty probe only)", log_file)
        log("="*80 + "\n", log_file)

        # Get GPU name
        device = get_gpu_name()
        log(f"GPU: {device}\n", log_file)

        # Run benchmarks
        results = {}
        results['baseline'] = run_baseline(bench_dir, iterations, log_file)
        log("", log_file)

        if comprehensive:
            # Run all BPF test modes
            bpf_modes = [
                ('bpf_empty', 'empty'),
                ('bpf_entry', 'entry'),
                ('bpf_exit', 'exit'),
                ('bpf_both', 'both'),
                ('bpf_ringbuf', 'ringbuf'),
                ('bpf_array_update', 'array-update'),
                ('bpf_array_lookup', 'array-lookup'),
                ('bpf_hash_update', 'hash-update'),
                ('bpf_hash_lookup', 'hash-lookup'),
                ('bpf_hash_delete', 'hash-delete'),
                ('bpf_percpu_array_update', 'percpu-array-update'),
                ('bpf_percpu_array_lookup', 'percpu-array-lookup'),
                ('bpf_percpu_hash_update', 'percpu-hash-update'),
                ('bpf_percpu_hash_lookup', 'percpu-hash-lookup'),
            ]

            for key, mode in bpf_modes:
                results[key] = run_bpf(bench_dir, build_dir, mode, iterations, log_file)
                log("", log_file)

            # Run NVBit
            results['nvbit'] = run_nvbit(bench_dir, iterations, log_file)
        else:
            # Quick mode - just empty probe
            results['bpf'] = run_bpf(bench_dir, build_dir, 'empty', iterations, log_file)
            log("", log_file)
            results['nvbit'] = run_nvbit(bench_dir, iterations, log_file)

        # Print results
        print_results_table(device, results, comprehensive, log_file)

        log(f"\nLog written to: {log_filename}", log_file)

if __name__ == '__main__':
    main()
