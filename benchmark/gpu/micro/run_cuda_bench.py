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
from typing import Optional, Dict

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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

def run_baseline(bench_dir: str, iterations: int = 10000) -> Optional[float]:
    """Run baseline benchmark without instrumentation."""
    print(f"{Colors.OKBLUE}Running baseline benchmark...{Colors.ENDC}")
    try:
        result = subprocess.run(
            [f'{bench_dir}/vec_add', str(iterations)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=bench_dir
        )
        if result.returncode != 0:
            print(f"{Colors.FAIL}Baseline failed: {result.stderr}{Colors.ENDC}")
            return None

        avg_time = parse_benchmark_output(result.stdout)
        if avg_time:
            print(f"{Colors.OKGREEN}✓ Baseline: {avg_time:.2f} μs{Colors.ENDC}")
        return avg_time
    except Exception as e:
        print(f"{Colors.FAIL}Error running baseline: {e}{Colors.ENDC}")
        return None

def run_bpf(bench_dir: str, build_dir: str, iterations: int = 10000) -> Optional[float]:
    """Run benchmark with BPF instrumentation."""
    print(f"{Colors.OKBLUE}Running BPF benchmark...{Colors.ENDC}")

    probe_process = None
    try:
        # Start BPF probe in background
        syscall_server = f"{build_dir}/runtime/syscall-server/libbpftime-syscall-server.so"
        agent = f"{build_dir}/runtime/agent/libbpftime-agent.so"

        if not os.path.exists(syscall_server):
            print(f"{Colors.FAIL}Error: {syscall_server} not found. Did you build bpftime?{Colors.ENDC}")
            return None
        if not os.path.exists(agent):
            print(f"{Colors.FAIL}Error: {agent} not found. Did you build bpftime?{Colors.ENDC}")
            return None

        probe_env = os.environ.copy()
        probe_env['BPFTIME_LOG_OUTPUT'] = 'console'
        probe_env['LD_PRELOAD'] = syscall_server

        probe_process = subprocess.Popen(
            [f'{bench_dir}/cuda_probe'],
            env=probe_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=bench_dir
        )

        # Wait for probe to initialize
        time.sleep(2)

        # Run benchmark with BPF agent
        agent_env = os.environ.copy()
        agent_env['BPFTIME_LOG_OUTPUT'] = 'console'
        agent_env['LD_PRELOAD'] = agent

        result = subprocess.run(
            [f'{bench_dir}/vec_add', str(iterations)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=bench_dir,
            env=agent_env
        )

        if result.returncode != 0:
            print(f"{Colors.FAIL}BPF benchmark failed: {result.stderr}{Colors.ENDC}")
            return None

        avg_time = parse_benchmark_output(result.stdout)
        if avg_time:
            print(f"{Colors.OKGREEN}✓ BPF: {avg_time:.2f} μs{Colors.ENDC}")
        return avg_time

    except Exception as e:
        print(f"{Colors.FAIL}Error running BPF benchmark: {e}{Colors.ENDC}")
        return None
    finally:
        if probe_process:
            try:
                probe_process.terminate()
                probe_process.wait(timeout=5)
            except:
                probe_process.kill()

def run_nvbit(bench_dir: str, iterations: int = 10000) -> Optional[float]:
    """Run benchmark with NVBit instrumentation."""
    print(f"{Colors.OKBLUE}Running NVBit benchmark...{Colors.ENDC}")

    nvbit_so = f"{bench_dir}/nvbit_vec_add.so"
    if not os.path.exists(nvbit_so):
        print(f"{Colors.FAIL}Error: {nvbit_so} not found. Did you run 'make' in benchmark/gpu?{Colors.ENDC}")
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

        if result.returncode != 0:
            print(f"{Colors.FAIL}NVBit benchmark failed: {result.stderr}{Colors.ENDC}")
            return None

        avg_time = parse_benchmark_output(result.stdout)
        if avg_time:
            print(f"{Colors.OKGREEN}✓ NVBit: {avg_time:.2f} μs{Colors.ENDC}")
        return avg_time

    except Exception as e:
        print(f"{Colors.FAIL}Error running NVBit benchmark: {e}{Colors.ENDC}")
        return None

def print_results_table(device: str, results: Dict[str, Optional[float]]):
    """Print the results in a formatted table."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}CUDA Vector Addition Benchmark Results{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

    baseline = results.get('baseline')
    bpf = results.get('bpf')
    nvbit = results.get('nvbit')

    print(f"{Colors.BOLD}Device: {Colors.OKCYAN}{device}{Colors.ENDC}\n")

    # Table header
    print(f"{Colors.BOLD}{'Method':<20} {'Avg Time (μs)':<20} {'Overhead':<15}{Colors.ENDC}")
    print(f"{'-'*55}")

    # Baseline
    if baseline:
        print(f"{'Baseline':<20} {baseline:<20.2f} {'-':<15}")
    else:
        print(f"{'Baseline':<20} {'FAILED':<20} {'-':<15}")

    # BPF
    if bpf and baseline:
        overhead = f"{bpf/baseline:.2f}x"
        print(f"{'BPF (bpftime)':<20} {bpf:<20.2f} {overhead:<15}")
    elif bpf:
        print(f"{'BPF (bpftime)':<20} {bpf:<20.2f} {'-':<15}")
    else:
        print(f"{'BPF (bpftime)':<20} {'FAILED':<20} {'-':<15}")

    # NVBit
    if nvbit and baseline:
        overhead = f"{nvbit/baseline:.2f}x"
        print(f"{'NVBit':<20} {nvbit:<20.2f} {overhead:<15}")
    elif nvbit:
        print(f"{'NVBit':<20} {nvbit:<20.2f} {'-':<15}")
    else:
        print(f"{'NVBit':<20} {'FAILED':<20} {'-':<15}")

    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

    # Print markdown table for easy copy-paste
    print(f"{Colors.BOLD}Markdown Table:{Colors.ENDC}")
    print(f"```markdown")
    print(f"| Device | Baseline | BPF (bpftime) | NVBit |")
    print(f"|--------|----------|---------------|-------|")

    bpf_str = f"{bpf:.1f} μs ({bpf/baseline:.2f}x)" if bpf and baseline else "FAILED"
    nvbit_str = f"{nvbit:.1f} μs ({nvbit/baseline:.2f}x)" if nvbit and baseline else "FAILED"
    baseline_str = f"{baseline:.1f} μs" if baseline else "FAILED"

    print(f"| **{device}** | {baseline_str} | {bpf_str} | {nvbit_str} |")
    print(f"```\n")

def main():
    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    bench_dir = script_dir
    build_dir = os.path.join(repo_root, 'build')

    if not os.path.exists(bench_dir):
        print(f"{Colors.FAIL}Error: benchmark/gpu directory not found{Colors.ENDC}")
        sys.exit(1)

    iterations = 10000
    if len(sys.argv) > 1:
        try:
            iterations = int(sys.argv[1])
        except ValueError:
            print(f"{Colors.WARNING}Invalid iterations argument, using default: {iterations}{Colors.ENDC}")

    print(f"{Colors.BOLD}{Colors.HEADER}Starting CUDA Benchmark Suite{Colors.ENDC}")
    print(f"{Colors.BOLD}Iterations: {iterations}{Colors.ENDC}\n")

    # Get GPU name
    device = get_gpu_name()

    # Run benchmarks
    results = {}
    results['baseline'] = run_baseline(bench_dir, iterations)
    print()
    results['bpf'] = run_bpf(bench_dir, build_dir, iterations)
    print()
    results['nvbit'] = run_nvbit(bench_dir, iterations)

    # Print results
    print_results_table(device, results)

if __name__ == '__main__':
    main()
