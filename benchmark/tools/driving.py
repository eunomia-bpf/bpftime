import re
import json
import numpy as np
import subprocess
import os
import signal
import time

def run_command(cmd):
    """Run a command in the background and return its process."""
    process = subprocess.Popen(cmd, shell=True)
    return process


def kill_process(process):
    """Kill a given process."""
    os.kill(process.pid, signal.SIGKILL)
    print("Process killed")
       # Give the process some time to terminate.
    time.sleep(1)

    # Check if the process has really terminated. If it has, poll() should return the exit code.
    if process.poll() is None:
        print(f"Process {process.pid} was not killed, forcing kill.")
        os.kill(process.pid, signal.SIGKILL)
    else:
        print(f"Process {process.pid} was successfully killed.")


# Function to run the command and extract the average write time
def run_command_and_extract_time(name: str, library: str):
    print("run_command_and_extract_time")
    try:
        result = subprocess.check_output(
            [
                "sudo",
                library,
                name,
            ],
            universal_newlines=True,
        )
        match = re.search(r"Average time usage (\d+\.\d+)ns,", result)
        print(float(match.group(1)))
        if match:
            return float(match.group(1))
        else:
            print("Warning: No match found in the output")
            return None
    except Exception as e:
        print(f"Error during command execution: {e}")
        return None


def save_micro_benchmark_data(name: str, library: str, output_file: str):
    # Run the command 100 times and collect the average write times
    times = [run_command_and_extract_time(name, library) for _ in range(20)]
    times = [time for time in times if time is not None]  # Filter out None values

    # Compute metrics
    mean_time = np.mean(times)
    median_time = np.median(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_dev_time = np.std(times)

    # Prepare the data for the JSON file
    data = {
        "raw_times": times,
        "mean": mean_time,
        "median": median_time,
        "min": min_time,
        "max": max_time,
        "std_dev": std_dev_time,
    }

    # Save the data to a JSON file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

def run_kernel_syscall_tracepoint_test():
    server = run_command("sudo benchmark/syscall/syscall")
    save_micro_benchmark_data(
        "benchmark/syscall/victim", "A=B", "benchmark/micro-bench/kernel-syscall.json"
    )
    kill_process(server)
    run_command("pkill syscall/syscall")

def run_userspace_syscall_tracepoint_test():
    server = run_command(
        "sudo LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so  benchmark/syscall/syscall"
    )
    save_micro_benchmark_data(
        "benchmark/syscall/victim",
        "LD_PRELOAD=build/runtime/agent/libbpftime-agent.so",
        "benchmark/micro-bench/userspace-syscall.json",
    )
    kill_process(server)
    run_command("pkill syscall/syscall")

def run_syscall_baseline_test():
    save_micro_benchmark_data(
        "benchmark/syscall/victim",
        "LD_PRELOAD=build/runtime/agent/libbpftime-agent.so",
        "benchmark/micro-bench/baseline-syscall.json",
    )


run_syscall_baseline_test()