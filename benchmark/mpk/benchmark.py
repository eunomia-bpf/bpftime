import asyncio
import subprocess
import re
import statistics
import time
import pathlib
import json
import os
import sys
import platform
from datetime import datetime
import math
import typing
import argparse

# Set up paths to work from project root
PROJECT_ROOT = pathlib.Path(os.getcwd())
# If we're already in the bpftime directory, don't add it again
if PROJECT_ROOT.name == "bpftime":
    BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
else:
    BENCHMARK_DIR = PROJECT_ROOT / "bpftime" / "benchmark"
MPK_DIR = BENCHMARK_DIR / "mpk"

# Set up logging
benchmark_logs = []

def log_message(msg: str):
    """Log a message with timestamp to both console and log collection"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    benchmark_logs.append(formatted_msg)

def extract_times(output):
    """Extract the benchmark times from command output."""
    times = []
    for line in output.decode().split('\n'):
        if 'Average time usage' in line:
            # Extract the number before 'ns'
            match = re.search(r'Average time usage (\d+\.\d+)', line)
            if match:
                time = float(match.group(1))
                times.append(time)
    return times

def parse_victim_output(v: str) -> dict:
    """Parse output from the test binary into a dictionary of test results."""
    lines = v.strip().splitlines()
    i = 0
    result = {}
    while i < len(lines):
        curr_line = lines[i]
        if curr_line.startswith("Benchmarking"):
            name = curr_line.split()[1]
            time_usage = float((lines[i + 1]).split()[3])
            result[name] = time_usage
            i += 2
        else:
            i += 1
    return result

async def handle_stdout(
    stdout: asyncio.StreamReader,
    notify: asyncio.Event,
    title: str,
    callback: typing.Optional[typing.Tuple[asyncio.Event, str]] = None,
    log_output: bool = True,
):
    """Handle process stdout, watching for a signal if needed."""
    output_lines = []
    while True:
        t1 = asyncio.create_task(notify.wait())
        t2 = asyncio.create_task(stdout.readline())
        done, pending = await asyncio.wait(
            [t1, t2],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for item in pending:
            item.cancel()
        if t2 in done:
            s = t2.result().decode()
            if not s:  # Skip empty lines
                continue
            print(f"{title}: {s}", end="")
            if log_output:
                output_lines.append(f"{title}: {s.strip()}")
            if callback:
                evt, sig = callback
                if sig in s:
                    evt.set()
                    log_message(f"Callback triggered: {sig}")
        if t1 in done:
            break
        if stdout.at_eof():
            break
    return output_lines

def handle_result(result: dict) -> dict:
    """Process raw timing results into a structured format with statistics."""
    new_result = {}
    for k, v in result.items():
        if not v:  # Skip empty results
            continue
            
        avg = sum(v) / len(v)
        sqr_sum = sum(x**2 for x in v)
        sqr_diff = sqr_sum / len(v) - (avg**2)
        std_dev = math.sqrt(sqr_diff) if sqr_diff > 0 else 0
        
        new_result[k] = {
            "max": max(v),
            "min": min(v),
            "avg": avg,
            "std_dev": std_dev,
            "raw_values": v
        }
        
    return new_result

async def run_mpk_test(num_runs=10):
    """Run benchmark with MPK enabled."""
    log_message(f"Starting MPK-enabled test with {num_runs} runs")
    should_exit = asyncio.Event()
    server_start_cb = asyncio.Event()
    
    log_message("Launching MPK-enabled server process")
    
    # Add environment variable for MPK server
    env = {
        "LD_PRELOAD": str(PROJECT_ROOT / "build-mpk/runtime/syscall-server/libbpftime-syscall-server.so"),
        "SPDLOG_LEVEL": "debug"  # Enable more detailed debug output
    }
    
    # Use generic 'uprobe' server - same server is used for both MPK and non-MPK
    server = await asyncio.subprocess.create_subprocess_exec(
        str(BENCHMARK_DIR / "uprobe/uprobe"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=PROJECT_ROOT,
        env=env,
    )
    
    server_stdout_task = asyncio.get_running_loop().create_task(
        handle_stdout(
            server.stdout,
            should_exit,
            "SERVER",
            (server_start_cb, "Successfully started!"),
        )
    )
    
    try:
        # Wait for server to start with timeout
        await asyncio.wait_for(server_start_cb.wait(), timeout=10.0)
        log_message("MPK server started successfully")
        
        # Wait 2 seconds after server starts before running tests
        log_message("Waiting 2 seconds before starting tests...")
        await asyncio.sleep(2)
        log_message("Delay completed, starting tests now")
    except asyncio.TimeoutError:
        log_message("WARNING: Timeout waiting for server to start, proceeding anyway")
        # Set the event to prevent hanging
        server_start_cb.set()
    
    result = {}
    
    for i in range(num_runs):
        log_message(f"Starting MPK-enabled run {i+1}/{num_runs}")
        victim = await asyncio.subprocess.create_subprocess_shell(
            str(BENCHMARK_DIR / "test"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            env={
                "LD_PRELOAD": str(PROJECT_ROOT / "build-mpk/runtime/agent/libbpftime-agent.so"),
                "BPFTIME_LOG_OUTPUT": "console"
            },
        )
        victim_out, _ = await victim.communicate()
        victim_out_str = victim_out.decode()
        log_message(f"MPK-enabled output for run {i+1}:\n{victim_out_str}")
        
        curr = parse_victim_output(victim_out_str)
        if not result:
            result = {k: [v] for k, v in curr.items()}
        else:
            for k, v in curr.items():
                if k in result:
                    result[k].append(v)
                else:
                    result[k] = [v]
        log_message(f"MPK-enabled run {i+1} results: {curr}")
    
    should_exit.set()
    await server_stdout_task
    log_message("Terminating MPK server")
    server.kill()
    await server.communicate()
    
    processed_result = handle_result(result)
    log_message(f"MPK-enabled test completed with results: {processed_result}")
    return processed_result

async def run_standard_test(num_runs=10):
    """Run benchmark with standard (non-MPK) configuration."""
    log_message(f"Starting standard (non-MPK) test with {num_runs} runs")
    should_exit = asyncio.Event()
    server_start_cb = asyncio.Event()
    
    log_message("Launching standard server process")
    
    # Add environment variable for standard server
    env = {
        "LD_PRELOAD": str(PROJECT_ROOT / "build/runtime/syscall-server/libbpftime-syscall-server.so"),
        "SPDLOG_LEVEL": "debug"  # Enable more detailed debug output
    }
    
    # Use generic 'uprobe' server - same server is used for both MPK and non-MPK
    server = await asyncio.subprocess.create_subprocess_exec(
        str(BENCHMARK_DIR / "uprobe/uprobe"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=PROJECT_ROOT,
        env=env,
    )
    
    server_stdout_task = asyncio.get_running_loop().create_task(
        handle_stdout(
            server.stdout,
            should_exit,
            "SERVER",
            (server_start_cb, "Successfully started!"),
        )
    )
    
    try:
        # Wait for server to start with timeout
        await asyncio.wait_for(server_start_cb.wait(), timeout=10.0)
        log_message("Standard server started successfully")
        
        # Wait 2 seconds after server starts before running tests
        log_message("Waiting 2 seconds before starting tests...")
        await asyncio.sleep(2)
        log_message("Delay completed, starting tests now")
    except asyncio.TimeoutError:
        log_message("WARNING: Timeout waiting for server to start, proceeding anyway")
        # Set the event to prevent hanging
        server_start_cb.set()
    
    result = {}
    
    for i in range(num_runs):
        log_message(f"Starting standard run {i+1}/{num_runs}")
        victim = await asyncio.subprocess.create_subprocess_shell(
            str(BENCHMARK_DIR / "test"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            env={
                "LD_PRELOAD": str(PROJECT_ROOT / "build/runtime/agent/libbpftime-agent.so"),
                "BPFTIME_LOG_OUTPUT": "console"
            },
        )
        victim_out, _ = await victim.communicate()
        victim_out_str = victim_out.decode()
        log_message(f"Standard output for run {i+1}:\n{victim_out_str}")
        
        curr = parse_victim_output(victim_out_str)
        if not result:
            result = {k: [v] for k, v in curr.items()}
        else:
            for k, v in curr.items():
                if k in result:
                    result[k].append(v)
                else:
                    result[k] = [v]
        log_message(f"Standard run {i+1} results: {curr}")
    
    should_exit.set()
    await server_stdout_task
    log_message("Terminating standard server")
    server.kill()
    await server.communicate()
    
    processed_result = handle_result(result)
    log_message(f"Standard test completed with results: {processed_result}")
    return processed_result

def generate_markdown_report(results, output_path):
    """Generate a markdown report from benchmark results."""
    log_message(f"Generating markdown report at {output_path}")
    
    # Get system information
    def get_cpu_info():
        cpu_info = {"model": "Unknown", "cores": "Unknown", "threads": "Unknown"}
        try:
            # Try to get CPU model name from /proc/cpuinfo on Linux
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    model_name = None
                    processor_count = 0
                    for line in f:
                        if line.startswith('model name'):
                            if not model_name:  # Take only the first model name
                                model_name = line.split(':')[1].strip()
                        if line.startswith('processor'):
                            processor_count += 1
                    
                    if model_name:
                        cpu_info['model'] = model_name
                    cpu_info['threads'] = processor_count
                    
                    # Try to get physical core count
                    try:
                        # Look for unique combinations of physical id and core id
                        physical_ids = set()
                        with open('/proc/cpuinfo', 'r') as f:
                            physical_id = None
                            core_id = None
                            cores = set()
                            for line in f:
                                if line.startswith('physical id'):
                                    physical_id = line.split(':')[1].strip()
                                elif line.startswith('core id'):
                                    core_id = line.split(':')[1].strip()
                                elif line.strip() == '':
                                    if physical_id is not None and core_id is not None:
                                        cores.add((physical_id, core_id))
                                    physical_id = None
                                    core_id = None
                            cpu_info['cores'] = len(cores)
                    except:
                        # Fallback if we can't get physical core count
                        cpu_info['cores'] = cpu_info['threads']
            else:
                # Use platform module as fallback
                cpu_info['model'] = platform.processor() or "Unknown"
                # No reliable way to get core count with standard library only
                cpu_info['cores'] = cpu_info['threads'] = os.cpu_count() or "Unknown"
                
            return cpu_info
        except Exception as e:
            log_message(f"Error getting CPU info: {e}")
            return cpu_info
    
    # Get memory info
    def get_memory_info():
        try:
            # Try to get memory info from /proc/meminfo on Linux
            if os.path.exists('/proc/meminfo'):
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            # Convert kB to GB
                            total_kb = int(line.split()[1])
                            return f"{total_kb / (1024**2):.2f} GB"
            # Fallback
            return "Unknown"
        except Exception as e:
            log_message(f"Error getting memory info: {e}")
            return "Unknown"
    
    # Format environment info
    cpu_info = get_cpu_info()
    env_info = {
        "OS": f"{platform.system()} {platform.release()}",
        "CPU": f"{cpu_info.get('model', 'Unknown')} ({cpu_info.get('cores', 'Unknown')} cores, {cpu_info.get('threads', 'Unknown')} threads)",
        "Memory": get_memory_info(),
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Python": platform.python_version(),
    }
    
    # Start building markdown content
    markdown = [
        "# BPFtime MPK Benchmark Results\n",
        f"*Generated on {env_info['Date']}*\n",
        "## Environment\n",
        f"- **OS:** {env_info['OS']}",
        f"- **CPU:** {env_info['CPU']}",
        f"- **Memory:** {env_info['Memory']}",
        f"- **Python:** {env_info['Python']}",
        "\n## Summary\n",
        "This benchmark compares two different userspace eBPF execution environments:",
        "- **MPK-enabled**: BPFtime with Memory Protection Keys (MPK) enabled",
        "- **Standard**: BPFtime without Memory Protection Keys",
        "\n*Times shown in nanoseconds (ns) - lower is better*\n"
    ]
    
    # Create summary comparison table
    markdown.extend([
        "### Performance Summary\n",
        "| Operation | MPK (ns) | Standard (ns) | Difference (ns) | Overhead |",
        "|-----------|----------|---------------|-----------------|----------|"
    ])
    
    mpk_results = results.get("mpk", {})
    standard_results = results.get("standard", {})
    
    for op in sorted(set(mpk_results.keys()) | set(standard_results.keys())):
        if op in mpk_results and op in standard_results:
            mpk_avg = mpk_results[op]["avg"]
            std_avg = standard_results[op]["avg"]
            diff = mpk_avg - std_avg
            overhead = (diff / std_avg) * 100 if std_avg > 0 else float('inf')
            
            markdown.append(f"| {op} | {mpk_avg:.2f} | {std_avg:.2f} | {diff:+.2f} | {overhead:+.2f}% |")
    
    # Create detailed comparison table
    markdown.extend([
        "\n### Detailed Comparison\n",
        "| Operation | Environment | Min (ns) | Max (ns) | Avg (ns) | Std Dev |",
        "|-----------|-------------|----------|----------|----------|---------|"
    ])
    
    # Add data for each operation from both environments
    for op in sorted(set(mpk_results.keys()) | set(standard_results.keys())):
        for env_dict, env_name in [(mpk_results, "MPK"), (standard_results, "Standard")]:
            if op in env_dict:
                metrics = env_dict[op]
                min_val = f"{metrics['min']:.2f}"
                max_val = f"{metrics['max']:.2f}"
                avg_val = f"{metrics['avg']:.2f}"
                std_dev = f"{metrics['std_dev']:.2f}"
                
                markdown.append(f"| {op} | {env_name} | {min_val} | {max_val} | {avg_val} | {std_dev} |")
    
    # Add benchmark metadata
    benchmark_info = results.get("benchmark_info", {})
    markdown.extend([
        "\n## Benchmark Metadata\n",
        f"- **Number of runs:** {benchmark_info.get('num_runs', 'Unknown')}",
        f"- **Timestamp:** {benchmark_info.get('timestamp', 'Unknown')}",
        f"- **Total duration:** {benchmark_info.get('total_duration_seconds', 0):.2f} seconds\n"
    ])
    
    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(markdown))
    
    log_message(f"Markdown report generated successfully at {output_path}")

async def main():
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Run bpftime MPK benchmarks')
    parser.add_argument('--iter', type=int, default=10, help='Number of iterations for each benchmark test (default: 10)')
    args = parser.parse_args()

    start_time = time.time()
    log_message("Starting MPK benchmark suite")
    log_message(f"Project root: {PROJECT_ROOT}")
    log_message(f"Benchmark directory: {BENCHMARK_DIR}")
    log_message(f"MPK directory: {MPK_DIR}")
    
    # Number of iterations - use command line argument instead of hardcoded value
    iterations = args.iter
    log_message(f"Will run each benchmark {iterations} times")
    
    try:
        # Run benchmarks
        log_message("Running MPK-enabled benchmarks")
        mpk_results = await run_mpk_test(iterations)
        
        log_message("Running standard (non-MPK) benchmarks")
        standard_results = await run_standard_test(iterations)
        
        # Combine results
        results = {
            "mpk": mpk_results,
            "standard": standard_results,
            "benchmark_info": {
                "num_runs": iterations,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_duration_seconds": time.time() - start_time
            }
        }
        
        # Save results to JSON file
        json_output_path = MPK_DIR / "benchmark-output.json"
        log_message(f"Saving benchmark results to {json_output_path}")
        with open(json_output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate Markdown report
        markdown_output_path = MPK_DIR / "results.md"
        generate_markdown_report(results, markdown_output_path)
        
        # Save logs to text file
        log_output_path = MPK_DIR / "benchmark-logs.txt"
        log_message(f"Saving benchmark logs to {log_output_path}")
        with open(log_output_path, "w") as f:
            f.write("\n".join(benchmark_logs))
        
        # Print summary
        log_message("Benchmark Summary:")
        for test_type, test_results in results.items():
            if test_type == "benchmark_info":
                continue
            log_message(f"  {test_type}:")
            for operation, metrics in test_results.items():
                log_message(f"    {operation}: avg={metrics['avg']:.4f}, min={metrics['min']:.4f}, max={metrics['max']:.4f}, std_dev={metrics['std_dev']:.4f}")
        
        log_message(f"Benchmark completed in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        error_msg = f"ERROR: Benchmark failed with exception: {str(e)}"
        log_message(error_msg)
        
        # Save logs even on failure
        log_output_path = MPK_DIR / "benchmark-logs-error.txt"
        with open(log_output_path, "w") as f:
            f.write("\n".join(benchmark_logs))
        
        print(f"Logs saved to {log_output_path}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 