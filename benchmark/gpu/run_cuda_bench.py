#!/usr/bin/env python3
"""
CUDA Benchmark Runner for bpftime
Runs benchmarks using JSON configuration.

IMPORTANT: This script must be run from the bpftime project root directory.
Usage: python3 benchmark/gpu/run_cuda_bench.py <config.json>
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
DEFAULT_CONFIG = "benchmark/gpu/micro/micro_vec_add_config.json"


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
        cmd = [f'{bench_dir}/vec_add'] + \
            (vec_add_args.split() if vec_add_args else [])
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
            log(
                f"Baseline failed with return code {result.returncode}", log_file)
            return None

        avg_time = parse_benchmark_output(result.stdout)
        if avg_time:
            log(f"✓ Baseline: {avg_time:.2f} μs", log_file)
        return avg_time
    except Exception as e:
        log(f"Error running baseline: {e}", log_file)
        return None


def resolve_env_paths(env_dict: Dict[str, str], repo_root: str) -> Dict[str, str]:
    """Resolve relative paths in environment variables."""
    resolved = {}
    for key, value in env_dict.items():
        # If it looks like a path and is relative, resolve it
        if ('/' in value or '\\' in value) and not os.path.isabs(value):
            resolved_path = os.path.join(repo_root, value)
            if os.path.exists(resolved_path):
                resolved[key] = resolved_path
            else:
                resolved[key] = value  # Use as-is if not found
        else:
            resolved[key] = value
    return resolved


def run_bpf_test(bench_dir: str, build_dir: str, probe_cmd: str, vec_add_args: str,
                 repo_root: str, log_file=None, vec_add_binary: str = None,
                 test_env: Dict[str, str] = None, global_env: Dict[str, str] = None,
                 probe_env_config: Dict[str, str] = None, agent_env_config: Dict[str, str] = None) -> Optional[float]:
    """Run benchmark with or without BPF instrumentation.

    Args:
        test_env: Environment variables for simple test runs (no probe)
        global_env: Global environment variables to apply to all runs
        probe_env_config: Environment config for probe process (bpftime mode)
        agent_env_config: Environment config for agent process (bpftime mode)
    """

    # Use vec_add_binary if provided, otherwise raise error
    if vec_add_binary:
        vec_add_cmd_base = vec_add_binary
        vec_add_dir = os.path.dirname(vec_add_binary)
    else:
        raise RuntimeError("vec_add_binary must be provided")

    # Special case: empty probe_cmd means no eBPF at all (true baseline)
    # Can still have test_env for simple LD_PRELOAD cases
    if not probe_cmd:
        if test_env or global_env:
            log(
                f"Running test with custom env: vec_add_binary={vec_add_cmd_base}, args={vec_add_args}", log_file)
        else:
            log(
                f"Running baseline (no eBPF): vec_add_binary={vec_add_cmd_base}, args={vec_add_args}", log_file)

        try:
            # Prepare environment: start with system env, apply global, then test-specific
            run_env = os.environ.copy()

            if global_env:
                resolved_global = resolve_env_paths(global_env, repo_root)
                run_env.update(resolved_global)

            if test_env:
                resolved_test = resolve_env_paths(test_env, repo_root)
                run_env.update(resolved_test)

            if test_env or global_env:
                log(f"Environment variables: {test_env or global_env}", log_file)

            cmd = [vec_add_cmd_base] + \
                (vec_add_args.split() if vec_add_args else [])
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=vec_add_dir,
                env=run_env
            )
            log(f"STDOUT:\n{result.stdout}", log_file)
            log(f"STDERR:\n{result.stderr}", log_file)

            if result.returncode != 0:
                log(f"Test failed with return code {result.returncode}", log_file)
                return None

            avg_time = parse_benchmark_output(result.stdout)
            if avg_time:
                test_type = "Test with custom env" if test_env else "Baseline"
                log(f"✓ {test_type}: {avg_time:.2f} μs", log_file)
            return avg_time
        except Exception as e:
            log(f"Error running baseline: {e}", log_file)
            return None

    # Run with BPF instrumentation
    log(f"Running BPF test: probe_cmd={probe_cmd}, vec_add_args={vec_add_args}", log_file)

    probe_process = None
    probe_log = f"{vec_add_dir}/.cuda_probe.stderr"
    try:
        # Prepare probe environment: system env + global + probe-specific config
        probe_env = os.environ.copy()

        if global_env:
            resolved_global = resolve_env_paths(global_env, repo_root)
            probe_env.update(resolved_global)

        if probe_env_config:
            resolved_probe = resolve_env_paths(probe_env_config, repo_root)
            probe_env.update(resolved_probe)
            log(f"Probe environment: {probe_env_config}", log_file)

        # Parse probe command (can include arguments)
        probe_cmd_parts = probe_cmd.split()
        probe_binary = os.path.join(repo_root, probe_cmd_parts[0])
        probe_args = probe_cmd_parts[1:] if len(probe_cmd_parts) > 1 else []

        if not os.path.exists(probe_binary):
            log(f"Error: Probe binary not found: {probe_binary}", log_file)
            return None

        probe_full_cmd = [probe_binary] + probe_args
        probe_stderr_file = open(probe_log, 'w')
        probe_process = subprocess.Popen(
            probe_full_cmd,
            env=probe_env,
            stdout=probe_stderr_file,
            stderr=probe_stderr_file,
            cwd=repo_root
        )

        # Wait for probe to initialize
        time.sleep(10)

        # Log probe startup (don't close the file yet, keep it open for continuous logging)
        probe_stderr_file.flush()
        if os.path.exists(probe_log):
            with open(probe_log, 'r') as f:
                probe_output = f.read()
                log(f"Probe startup log:\n{probe_output}", log_file)

        # Prepare agent environment: system env + global + agent-specific config
        agent_env = os.environ.copy()

        if global_env:
            resolved_global = resolve_env_paths(global_env, repo_root)
            agent_env.update(resolved_global)

        if agent_env_config:
            resolved_agent = resolve_env_paths(agent_env_config, repo_root)
            agent_env.update(resolved_agent)
            log(f"Agent environment: {agent_env_config}", log_file)

        # Use vec_add binary (full path)
        vec_add_cmd = [vec_add_cmd_base] + \
            (vec_add_args.split() if vec_add_args else [])
        result = subprocess.run(
            vec_add_cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=vec_add_dir,
            env=agent_env
        )

        log(f"STDOUT:\n{result.stdout}", log_file)
        log(f"STDERR:\n{result.stderr}", log_file)

        if result.returncode != 0 and result.returncode != -11:  # Ignore Segmentation fault
            log(
                f"BPF benchmark failed with return code {result.returncode}", log_file)
            return None

        avg_time = parse_benchmark_output(result.stdout)
        if avg_time:
            log(f"✓ BPF test: {avg_time:.2f} μs", log_file)

        return avg_time

    except Exception as e:
        log(f"Error running BPF benchmark: {e}", log_file)
        return None
    finally:
        # Stop probe process and capture remaining output
        if probe_process:
            try:
                # Send SIGTERM to allow graceful shutdown
                probe_process.terminate()
                # Wait for it to finish and capture any remaining output
                probe_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log("Probe process didn't terminate gracefully, killing it", log_file)
                probe_process.kill()
                probe_process.wait()
            except Exception as e:
                log(f"Error terminating probe: {e}", log_file)
                try:
                    probe_process.kill()
                except:
                    pass

        # Close the probe log file handle
        try:
            if 'probe_stderr_file' in locals():
                probe_stderr_file.flush()
                probe_stderr_file.close()
        except:
            pass

        # Log all probe output (including what was printed during execution)
        if os.path.exists(probe_log):
            try:
                with open(probe_log, 'r') as f:
                    full_probe_output = f.read()
                    log(
                        f"\n=== Complete Probe Output ===\n{full_probe_output}\n=== End Probe Output ===\n", log_file)
            except Exception as e:
                log(f"Error reading probe log: {e}", log_file)

            # Clean up probe log file
            try:
                os.remove(probe_log)
            except:
                pass


def load_config(config_path: str, repo_root: str) -> Dict[str, Any]:
    """Load benchmark configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Resolve workload presets
    presets = config.get('workload_presets', {})
    for test_case in config.get('test_cases', []):
        workload = test_case.get('workload')
        if workload and workload in presets:
            # Get the preset which now includes full binary path and args
            preset_value = presets[workload]
            # Split into binary path and args
            parts = preset_value.split(' ', 1)
            if len(parts) == 2:
                binary_rel_path, args = parts
                # Convert relative path to absolute
                binary_path = os.path.join(repo_root, binary_rel_path)
                test_case['vec_add_args'] = args
                test_case['vec_add_binary'] = binary_path
        # Keep existing vec_add_args if no workload specified

    return config


def save_results(results: Dict[str, Any], output_path: str):
    """Save benchmark results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def get_workload_description(workload_name: str, presets: Dict[str, str]) -> str:
    """Get human-readable description of workload."""
    if workload_name not in presets:
        return workload_name

    args = presets[workload_name].split()
    # Format: "binary_path elements iterations threads blocks"
    if len(args) >= 5:
        elements, iterations, threads, blocks = args[1], args[2], args[3], args[4]
        return f"{elements} elements, {iterations} iterations, {threads} threads × {blocks} blocks"
    return workload_name


def print_results_table(results: Dict[str, Any], log_file=None):
    """Print the results in markdown format."""
    config = results.get('config', {})
    presets = config.get('workload_presets', {})

    # Markdown header
    log("\n# CUDA Benchmark Results\n", log_file)
    log(f"**Device:** {results.get('device', 'Unknown')}  ", log_file)
    log(f"**Timestamp:** {results.get('timestamp', 'Unknown')}  \n", log_file)

    # Workload configuration section
    log("## Workload Configuration\n", log_file)
    log("| Workload | Binary | Elements | Iterations | Threads | Blocks |", log_file)
    log("|----------|--------|----------|------------|---------|--------|", log_file)
    for name, args in sorted(presets.items()):
        parts = args.split()
        # Format: "binary_path elements iterations threads blocks"
        if len(parts) >= 5:
            log(f"| {name} | {parts[0]} | {parts[1]} | {parts[2]} | {parts[3]} | {parts[4]} |", log_file)
    log("", log_file)

    # Build baseline lookup table by name
    baseline_map = {}
    for test in results.get('tests', []):
        name = test['name']
        if name.startswith('Baseline'):
            baseline_map[name] = test.get('avg_time_us')

    # Benchmark results table
    log("## Benchmark Results\n", log_file)
    log("| Test Name | Workload | Avg Time (μs) | vs Baseline | Overhead |", log_file)
    log("|-----------|----------|---------------|-------------|----------|", log_file)

    # Test cases
    for test in results.get('tests', []):
        name = test['name']
        avg_time = test.get('avg_time_us')
        baseline_ref = test.get('baseline')
        workload = test.get('workload', '-')

        # Get the baseline time for this specific test
        baseline_time = None
        if baseline_ref and baseline_ref in baseline_map:
            baseline_time = baseline_map[baseline_ref]

        if avg_time and baseline_time:
            multiplier = avg_time / baseline_time
            overhead_pct = (multiplier - 1) * 100
            overhead_str = f"{multiplier:.2f}x" if multiplier >= 1 else f"0.{int(multiplier*100)}x"
            overhead_pct_str = f"+{overhead_pct:.1f}%" if overhead_pct > 0 else f"{overhead_pct:.1f}%"
            log(f"| {name} | {workload} | {avg_time:.2f} | {baseline_time:.2f} | {overhead_str} ({overhead_pct_str}) |", log_file)
        elif avg_time:
            log(f"| {name} | {workload} | {avg_time:.2f} | - | - |", log_file)
        else:
            log(f"| {name} | {workload} | FAILED | - | - |", log_file)

    log("", log_file)


def main():
    # Determine paths - assume running from repo root
    repo_root = os.getcwd()
    build_dir = os.path.join(repo_root, 'build')

    # Parse arguments
    config_file = DEFAULT_CONFIG
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    # Config file path - if it's a full path, use it; otherwise relative to repo root
    if os.path.isabs(config_file):
        config_path = config_file
    else:
        # Relative to repo root
        config_path = os.path.join(repo_root, config_file)
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        print(f"Usage: {sys.argv[0]} [config.json]")
        sys.exit(1)

    # Load configuration
    try:
        config = load_config(config_path, repo_root)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Extract config
    test_cases = config.get('test_cases', [])

    # Get output path prefix from config (should be relative to repo root)
    output_prefix = config.get('output_prefix', 'benchmark/gpu/result')

    # Convert output_prefix to absolute path from repo root
    if not os.path.isabs(output_prefix):
        output_dir = os.path.dirname(os.path.join(repo_root, output_prefix))
        output_basename = os.path.basename(output_prefix)
    else:
        output_dir = os.path.dirname(output_prefix)
        output_basename = os.path.basename(output_prefix)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct output file paths
    log_file_name = f"{output_basename}_bench.log"
    result_file_name = f"{output_basename}_result.json"
    result_md_file_name = f"{output_basename}_result.md"

    log_path = os.path.join(output_dir, log_file_name)
    result_path = os.path.join(output_dir, result_file_name)
    result_md_path = os.path.join(output_dir, result_md_file_name)

    with open(log_path, 'w') as log_file:
        log("="*80, log_file)
        log("CUDA Benchmark Suite", log_file)
        log("="*80, log_file)
        log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
        log(f"Config file: {config_file}", log_file)
        log("="*80 + "\n", log_file)

        # Get GPU name
        device = get_gpu_name()
        log(f"GPU: {device}\n", log_file)

        # Prepare results
        results = {
            'device': device,
            'timestamp': datetime.now().isoformat(),
            'config_file': config_file,
            'config': config,
            'tests': []
        }

        # Extract global and mode-specific environment configs
        global_env = config.get('global_env', {})
        probe_env_config = config.get('probe_env', {})
        agent_env_config = config.get('agent_env', {})

        # Run test cases
        for test_case in test_cases:
            name = test_case.get('name', 'Unknown')
            probe_binary_cmd = test_case.get('probe_binary_cmd', '')
            test_vec_add_args = test_case.get('vec_add_args', '')
            baseline_ref = test_case.get('baseline')
            vec_add_binary = test_case.get('vec_add_binary')
            # Optional per-test environment variables
            test_env = test_case.get('env')

            if not test_vec_add_args:
                log(f"Skipping {name}: no vec_add_args specified", log_file)
                continue

            log(f"Running test: {name}", log_file)
            avg_time = run_bpf_test(output_dir, build_dir, probe_binary_cmd, test_vec_add_args,
                                    repo_root, log_file, vec_add_binary, test_env, global_env,
                                    probe_env_config, agent_env_config)

            results['tests'].append({
                'name': name,
                'probe_binary_cmd': probe_binary_cmd,
                'vec_add_args': test_vec_add_args,
                'workload': test_case.get('workload', ''),
                'baseline': baseline_ref,
                'avg_time_us': avg_time
            })
            log("", log_file)

        # Save results
        save_results(results, result_path)
        log(f"Results saved to: {result_file_name}", log_file)

        # Print results table to log
        print_results_table(results, log_file)

        log(f"Log saved to: {log_file_name}", log_file)

    # Save markdown output
    with open(result_md_path, 'w') as md_file:
        print_results_table(results, md_file)

    print(f"\nMarkdown results saved to: {result_md_file_name}")
    print(f"JSON results saved to: {result_file_name}")
    print(f"Log saved to: {log_file_name}")


if __name__ == '__main__':
    main()
