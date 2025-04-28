import asyncio
import typing
import pathlib
import math
import json
import time
import os
import sys
import subprocess

# Fix the path structure to avoid duplicate bpftime directory
# Set up paths to work from project root
PROJECT_ROOT = pathlib.Path(os.getcwd())
# If we're already in the bpftime directory, don't add it again
if PROJECT_ROOT.name == "bpftime":
    BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
else:
    BENCHMARK_DIR = PROJECT_ROOT / "bpftime" / "benchmark"
UPROBE_DIR = BENCHMARK_DIR / "uprobe"

# Set up logging
benchmark_logs = []

def log_message(msg: str):
    """Log a message with timestamp to both console and log collection"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    benchmark_logs.append(formatted_msg)

def parse_victim_output(v: str) -> dict:
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
    new_result = {}
    for k, v in result.items():
        avg = sum(v) / len(v)
        sqr_sum = sum(x**2 for x in v)
        sqr_diff = sqr_sum / len(v) - (avg**2)
        std_dev = math.sqrt(sqr_diff) if sqr_diff > 0 else 0
        val = {
            "max": max(v), 
            "min": min(v), 
            "avg": avg, 
            "std_dev": std_dev,
            "raw_values": v
        }
        new_result[k] = val
    return new_result


async def run_userspace_uprobe_test(num_runs=10):
    log_message(f"Starting userspace uprobe test with {num_runs} runs")
    should_exit = asyncio.Event()
    server_start_cb = asyncio.Event()
    
    log_message(f"Launching uprobe server: {UPROBE_DIR / 'uprobe'}")
    server = await asyncio.subprocess.create_subprocess_exec(
        str(UPROBE_DIR / "uprobe"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=PROJECT_ROOT,
        env={
            "LD_PRELOAD": str(
                PROJECT_ROOT / "build/runtime/syscall-server/libbpftime-syscall-server.so"
            )
        },
    )
    
    server_stdout_task = asyncio.get_running_loop().create_task(
        handle_stdout(
            server.stdout,
            should_exit,
            "SERVER",
            (server_start_cb, "__bench_probe is for uprobe only"),
        )
    )
    
    await server_start_cb.wait()
    log_message("Server started successfully")
    
    result = None
    for i in range(num_runs):
        log_message(f"Starting userspace run {i+1}/{num_runs}")
        victim = await asyncio.subprocess.create_subprocess_shell(
            str(BENCHMARK_DIR / "test"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            env={
                "LD_PRELOAD": str(
                    PROJECT_ROOT / "build/runtime/agent/libbpftime-agent.so"
                )
            },
        )
        victim_out, _ = await victim.communicate()
        victim_out_str = victim_out.decode()
        log_message(f"Victim output for run {i+1}:\n{victim_out_str}")
        
        curr = parse_victim_output(victim_out_str)
        if result is None:
            result = {k: [v] for k, v in curr.items()}
        else:
            for k, v in curr.items():
                result[k].append(v)
        log_message(f"Run {i+1} results: {curr}")
    
    should_exit.set()
    await server_stdout_task
    log_message("Terminating server")
    server.kill()
    await server.communicate()
    
    processed_result = handle_result(result)
    log_message(f"Userspace uprobe test completed with results: {processed_result}")
    return processed_result


async def run_kernel_uprobe_test(num_runs=10):
    log_message(f"Starting kernel uprobe test with {num_runs} runs")
    should_exit = asyncio.Event()
    server_start_cb = asyncio.Event()
    
    log_message(f"Launching kernel uprobe server: {UPROBE_DIR / 'uprobe'}")
    server = await asyncio.subprocess.create_subprocess_exec(
        str(UPROBE_DIR / "uprobe"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=PROJECT_ROOT,
    )
    
    server_stdout_task = asyncio.get_running_loop().create_task(
        handle_stdout(
            server.stdout,
            should_exit,
            "SERVER",
            (server_start_cb, "__bench_probe is for uprobe only"),
        )
    )
    
    try:
        # Wait for server to start with timeout
        await asyncio.wait_for(server_start_cb.wait(), timeout=10.0)
        log_message("Kernel uprobe server started successfully")
    except asyncio.TimeoutError:
        log_message("WARNING: Timeout waiting for server to start, proceeding anyway")
        # Set the event to prevent hanging
        server_start_cb.set()
    
    result = None
    for i in range(num_runs):
        log_message(f"Starting kernel uprobe run {i+1}/{num_runs}")
        victim = await asyncio.subprocess.create_subprocess_shell(
            str(BENCHMARK_DIR / "test"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=PROJECT_ROOT,
        )
        victim_out, _ = await victim.communicate()
        victim_out_str = victim_out.decode()
        log_message(f"Victim output for run {i+1}:\n{victim_out_str}")
        
        curr = parse_victim_output(victim_out_str)
        if result is None:
            result = {k: [v] for k, v in curr.items()}
        else:
            for k, v in curr.items():
                result[k].append(v)
        log_message(f"Run {i+1} results: {curr}")
    
    should_exit.set()
    await server_stdout_task
    log_message("Terminating kernel uprobe server")
    server.kill()
    await server.communicate()
    
    processed_result = handle_result(result)
    log_message(f"Kernel uprobe test completed with results: {processed_result}")
    return processed_result


def handle_embed_victim_out(i: str) -> float:
    for line in i.splitlines():
        if line.startswith("avg function elapse time:"):
            return float(line.split()[-2])
    log_message("WARNING: Could not find avg function elapse time in output")
    return math.inf


async def run_embed_vm_test(num_runs=10):
    log_message(f"Starting embed VM test with {num_runs} runs")
    result = {"embed": []}
    bpf_path = str(UPROBE_DIR / ".output/uprobe.bpf.o")
    
    log_message(f"Using BPF object: {bpf_path}")
    
    for i in range(num_runs):
        log_message(f"Starting embed VM run {i+1}/{num_runs}")
        cmd = " ".join(
            [
                str(
                    PROJECT_ROOT / "build/benchmark/simple-benchmark-with-embed-ebpf-calling"
                ),
                bpf_path,
                bpf_path,
            ]
        )
        log_message(f"Running command: {cmd}")
        
        victim = await asyncio.subprocess.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=PROJECT_ROOT,
        )
        victim_out, _ = await victim.communicate()
        victim_out_str = victim_out.decode()
        log_message(f"Embed VM output for run {i+1}:\n{victim_out_str}")
        
        elapsed_time = handle_embed_victim_out(victim_out_str)
        result["embed"].append(elapsed_time)
        log_message(f"Run {i+1} embed VM elapsed time: {elapsed_time}")
    
    processed_result = handle_result(result)
    log_message(f"Embed VM test completed with results: {processed_result}")
    return processed_result


def ensure_sudo():
    """Re-run the script with sudo if not already running with privileges"""
    if os.geteuid() != 0:
        log_message("This benchmark requires sudo privileges. Re-launching with sudo...")
        args = ['sudo', sys.executable] + sys.argv
        # Exit the current process and start a new one with sudo
        os.execvp('sudo', args)
        # If execvp fails, the script will continue below
        log_message("Failed to re-launch with sudo. Some tests may fail.")
        return False
    return True


async def main():
    start_time = time.time()
    log_message("Starting uprobe benchmark suite")
    log_message(f"Project root: {PROJECT_ROOT}")
    log_message(f"Benchmark directory: {BENCHMARK_DIR}")
    log_message(f"Uprobe directory: {UPROBE_DIR}")
    
    log_message("=== Uprobe Benchmark Suite ===")
    log_message("Note: Kernel eBPF tests require sudo privileges.")
    log_message("")
    
    # Auto-elevate if needed before running kernel tests
    run_kernel = ensure_sudo()
    
    num_runs = 1
    log_message(f"Will run each benchmark {num_runs} times")

    try:
        if run_kernel:
            log_message("Starting kernel uprobe tests")
            k = await run_kernel_uprobe_test(num_runs)
        else:
            log_message("Skipping kernel uprobe tests (no sudo privileges)")
            k = {"skipped": "No sudo privileges"}
        
        log_message("Starting userspace uprobe tests")
        u = await run_userspace_uprobe_test(num_runs)
        
        log_message("Starting embedded VM tests")
        e = await run_embed_vm_test(num_runs)
        
        out = {
            "kernel_uprobe": k, 
            "userspace_uprobe": u, 
            "embed": e,
            "benchmark_info": {
                "num_runs": num_runs,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_duration_seconds": time.time() - start_time
            }
        }
        
        # Save results to JSON file
        json_output_path = UPROBE_DIR / "benchmark-output.json"
        log_message(f"Saving benchmark results to {json_output_path}")
        with open(json_output_path, "w") as f:
            json.dump(out, f, indent=2)
        
        # Save logs to text file
        log_output_path = UPROBE_DIR / "benchmark-logs.txt"
        log_message(f"Saving benchmark logs to {log_output_path}")
        with open(log_output_path, "w") as f:
            f.write("\n".join(benchmark_logs))
        
        # Print summary
        log_message("Benchmark Summary:")
        for test_type, results in out.items():
            if test_type == "benchmark_info":
                continue
            log_message(f"  {test_type}:")
            for metric, values in results.items():
                if isinstance(values, dict):
                    log_message(f"    {metric}: avg={values['avg']:.4f}, min={values['min']:.4f}, max={values['max']:.4f}, std_dev={values.get('std_dev', 0):.4f}")
        
        log_message(f"Benchmark completed in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        error_msg = f"ERROR: Benchmark failed with exception: {str(e)}"
        log_message(error_msg)
        
        # Save logs even on failure
        log_output_path = UPROBE_DIR / "benchmark-logs-error.txt"
        with open(log_output_path, "w") as f:
            f.write("\n".join(benchmark_logs))
        
        print(f"Logs saved to {log_output_path}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
