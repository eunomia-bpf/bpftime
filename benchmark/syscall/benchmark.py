#!/usr/bin/env python3
import subprocess
import re
import os
import time
import json
import statistics
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Configuration
NUM_RUNS = 10
VICTIM_PATH = "benchmark/syscall/victim"
SYSCALL_BPF_PATH = "benchmark/syscall/syscall"
AGENT_PATH = "build/runtime/agent/libbpftime-agent.so"
AGENT_TRANSFORMER_PATH = "build/attach/text_segment_transformer/libbpftime-agent-transformer.so"
SYSCALL_SERVER_PATH = "build/runtime/syscall-server/libbpftime-syscall-server.so"
BPFTIME_PATH = "~/.bpftime/bpftime"

# Result storage
results = {
    "native": [],               # No tracing
    "kernel_tracepoint": [],    # Kernel tracepoint syscall tracking
    "userspace_syscall": [],    # Userspace BPF syscall tracking
}

# Define a signal handler to prevent unexpected termination
def signal_handler(sig, frame):
    debug_print(f"Received signal {sig}, gracefully exiting...")
    cleanup_processes()
    sys.exit(0)

# Register the signal handler for common signals
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def debug_print(message):
    """Print debug messages with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[DEBUG {timestamp}] {message}")

def check_file_exists(path):
    """Check if a file exists and print its absolute path"""
    abs_path = os.path.abspath(path)
    exists = os.path.exists(abs_path)
    debug_print(f"Checking file: {abs_path} - {'EXISTS' if exists else 'NOT FOUND'}")
    return exists

def cleanup_processes():
    """Kill any running victim or syscall processes"""
    try:
        debug_print("Cleaning up processes...")
        
        # Get the PIDs of victim and syscall processes
        victim_pids = subprocess.run(["pgrep", "-f", "victim"], capture_output=True, text=True).stdout.strip().split()
        syscall_pids = subprocess.run(["pgrep", "-f", "syscall"], capture_output=True, text=True).stdout.strip().split()
        bpftime_pids = subprocess.run(["pgrep", "-f", "bpftime"], capture_output=True, text=True).stdout.strip().split()
        
        debug_print(f"Found victim PIDs: {victim_pids}")
        debug_print(f"Found syscall PIDs: {syscall_pids}")
        debug_print(f"Found bpftime PIDs: {bpftime_pids}")
        
        # Kill processes
        for pid_list, name in [(victim_pids, "victim"), 
                              (syscall_pids, "syscall"), 
                              (bpftime_pids, "bpftime")]:
            for pid in pid_list:
                try:
                    debug_print(f"Terminating {name} process with PID {pid}")
                    subprocess.run(["sudo", "kill", pid], stderr=subprocess.DEVNULL, check=False)
                except Exception as e:
                    debug_print(f"Error terminating {name} PID {pid}: {e}")
        
        # Wait for processes to terminate
        time.sleep(1)
        
        # Check if any processes are still running and try forceful termination if needed
        for pid_list, name in [(victim_pids, "victim"), 
                              (syscall_pids, "syscall"), 
                              (bpftime_pids, "bpftime")]:
            remaining_pids = subprocess.run(["pgrep", "-f", name], capture_output=True, text=True).stdout.strip().split()
            if remaining_pids:
                debug_print(f"Force killing remaining {name} processes: {remaining_pids}")
                for pid in remaining_pids:
                    subprocess.run(["sudo", "kill", "-9", pid], stderr=subprocess.DEVNULL, check=False)
    
    except Exception as e:
        debug_print(f"Error during cleanup: {e}")
        traceback.print_exc()

def parse_victim_output(output):
    """Parse victim output to extract average time usage"""
    debug_print(f"Parsing victim output: {output[:100]}...")  # Print first 100 chars
    match = re.search(r'Average time usage\s+(\d+\.\d+)ns', output)
    if match:
        return float(match.group(1))
    debug_print("Failed to parse victim output")
    return None

def run_native():
    """Run baseline benchmarks (no tracing)"""
    print("\n=== Running Native Tests (No Tracing) ===")
    
    try:
        for i in range(NUM_RUNS):
            print(f"Run {i+1}/{NUM_RUNS}...")
            
            # Run victim
            debug_print(f"Running victim: {VICTIM_PATH}")
            victim_cmd = [os.path.expanduser(VICTIM_PATH)]
            result = subprocess.run(victim_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                debug_print(f"victim failed with exit code {result.returncode}")
                debug_print(f"stdout: {result.stdout}")
                debug_print(f"stderr: {result.stderr}")
                continue
            
            avg_time = parse_victim_output(result.stdout)
            if avg_time:
                results["native"].append(avg_time)
                print(f"  Average time usage: {avg_time:.2f} ns")
            else:
                debug_print(f"Failed to parse output: {result.stdout}")
    
    except Exception as e:
        debug_print(f"Error in native test: {e}")
        traceback.print_exc()

def run_kernel_tracepoint():
    """Run kernel tracepoint benchmarks"""
    print("\n=== Running Kernel Tracepoint Tests ===")
    
    try:
        # Check for sudo
        if os.geteuid() != 0:
            debug_print("Kernel tracepoint tests require sudo. Please run with sudo.")
            return
        
        # Get victim PID first to target with tracepoint
        victim_pid = None
            
        # Start syscall BPF program
        debug_print(f"Starting syscall BPF program: {SYSCALL_BPF_PATH}")
        # Run the program directly since we're already using sudo for the script
        syscall_cmd = [os.path.expanduser(SYSCALL_BPF_PATH)]
        
        try:
            syscall_proc = subprocess.Popen(
                syscall_cmd, 
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            debug_print(f"Syscall BPF program started with PID: {syscall_proc.pid}")
            time.sleep(2)  # Give syscall time to start
            
            for i in range(NUM_RUNS):
                print(f"Run {i+1}/{NUM_RUNS}...")
                
                # Run victim
                debug_print(f"Running victim: {VICTIM_PATH}")
                victim_cmd = [os.path.expanduser(VICTIM_PATH)]
                result = subprocess.run(victim_cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    debug_print(f"victim failed with exit code {result.returncode}")
                    debug_print(f"stdout: {result.stdout}")
                    debug_print(f"stderr: {result.stderr}")
                    continue
                
                avg_time = parse_victim_output(result.stdout)
                if avg_time:
                    results["kernel_tracepoint"].append(avg_time)
                    print(f"  Average time usage: {avg_time:.2f} ns")
                else:
                    debug_print(f"Failed to parse output: {result.stdout}")
                
        except Exception as e:
            debug_print(f"Error in kernel tracepoint run: {e}")
            traceback.print_exc()
        finally:
            # Cleanup
            if 'syscall_proc' in locals() and syscall_proc:
                debug_print(f"Terminating syscall process PID: {syscall_proc.pid}")
                syscall_proc.terminate()
                try:
                    syscall_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    debug_print("Syscall process did not terminate, killing")
                    syscall_proc.kill()
    
    except Exception as e:
        debug_print(f"Error in kernel tracepoint test: {e}")
        traceback.print_exc()

def run_userspace_syscall():
    """Run userspace BPF syscall benchmarks"""
    print("\n=== Running Userspace BPF Syscall Tests ===")
    
    try:
        # Check for sudo
        if os.geteuid() != 0:
            debug_print("Userspace BPF tests require sudo. Please run with sudo.")
            return
            
        # Expand paths
        agent_path = os.path.abspath(AGENT_PATH)
        agent_transformer_path = os.path.abspath(AGENT_TRANSFORMER_PATH)
        syscall_server_path = os.path.abspath(SYSCALL_SERVER_PATH)
        bpftime_path = os.path.expanduser(BPFTIME_PATH)
        syscall_bpf_path = os.path.expanduser(SYSCALL_BPF_PATH)
        victim_path = os.path.expanduser(VICTIM_PATH)
        
        # Check if required files exist
        if not check_file_exists(agent_path) or not check_file_exists(syscall_server_path):
            debug_print("Skipping userspace BPF syscall tests: required agent or server files not found")
            return
            
        # Method 1: Use bpftime load/start approach (according to README.md)
        if check_file_exists(bpftime_path):
            try:
                debug_print("=== Method 1: Using bpftime command (from README.md) ===")
                # Load the syscall BPF program
                debug_print(f"Loading syscall BPF program with bpftime")
                load_cmd = [bpftime_path, "load", syscall_bpf_path]
                
                load_result = subprocess.run(load_cmd, capture_output=True, text=True, timeout=10)
                if load_result.returncode != 0:
                    debug_print(f"bpftime load failed with exit code {load_result.returncode}")
                    debug_print(f"stdout: {load_result.stdout}")
                    debug_print(f"stderr: {load_result.stderr}")
                    debug_print("Falling back to Method 2 (direct LD_PRELOAD)")
                    # Method 1 failed, try Method 2 below
                else:
                    debug_print("bpftime loaded successfully, waiting 2 seconds")
                    time.sleep(2)  # Give bpftime time to initialize
                    
                    for i in range(NUM_RUNS):
                        print(f"Run {i+1}/{NUM_RUNS}...")
                        
                        # Run victim with bpftime
                        debug_print(f"Running victim with bpftime: {victim_path}")
                        victim_cmd = [bpftime_path, "start", "-s", victim_path]
                        result = subprocess.run(victim_cmd, capture_output=True, text=True, timeout=60)
                        
                        if result.returncode != 0:
                            debug_print(f"bpftime start failed with exit code {result.returncode}")
                            debug_print(f"stdout: {result.stdout}")
                            debug_print(f"stderr: {result.stderr}")
                            continue
                        
                        avg_time = parse_victim_output(result.stdout)
                        if avg_time:
                            results["userspace_syscall"].append(avg_time)
                            print(f"  Average time usage: {avg_time:.2f} ns")
                        else:
                            debug_print(f"Failed to parse output: {result.stdout}")
                        
                    # Unload bpftime
                    debug_print("Unloading bpftime")
                    subprocess.run([bpftime_path, "unload"], 
                                stderr=subprocess.DEVNULL, 
                                stdout=subprocess.DEVNULL,
                                check=False)
                    
                    # If Method 1 succeeded, return early
                    if results["userspace_syscall"]:
                        return
                    
                    debug_print("Method 1 did not produce results, trying Method 2")
            
            except Exception as e:
                debug_print(f"Error in Method 1 (bpftime load/start): {e}")
                debug_print("Trying Method 2 (direct LD_PRELOAD)")
        else:
            debug_print("bpftime command not found, skipping Method 1")
        
        # Method 2: Direct LD_PRELOAD approach (according to README.md)
        debug_print("=== Method 2: Using direct LD_PRELOAD (from README.md) ===")
        
        try:
            # Check if transformer exists
            if not check_file_exists(agent_transformer_path):
                debug_print(f"Required agent transformer not found at {agent_transformer_path}")
                debug_print("Falling back to older method with just agent LD_PRELOAD")
                use_transformer = False
            else:
                use_transformer = True
            
            # First run syscall server with LD_PRELOAD
            debug_print(f"Starting syscall server with LD_PRELOAD={syscall_server_path}")
            env = os.environ.copy()
            env["LD_PRELOAD"] = syscall_server_path
            
            syscall_proc = subprocess.Popen(
                [os.path.expanduser(SYSCALL_BPF_PATH)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            debug_print(f"Syscall server started with PID: {syscall_proc.pid}")
            time.sleep(2)  # Give syscall server time to start
            
            for i in range(NUM_RUNS):
                print(f"Run {i+1}/{NUM_RUNS}...")
                
                # Run victim with agent LD_PRELOAD
                if use_transformer:
                    # According to README.md, newer method uses transformer and AGENT_SO
                    debug_print(f"Running victim with transformer LD_PRELOAD={agent_transformer_path}")
                    env = os.environ.copy()
                    env["AGENT_SO"] = agent_path
                    env["LD_PRELOAD"] = agent_transformer_path
                    debug_print(f"Environment: AGENT_SO={agent_path}, LD_PRELOAD={agent_transformer_path}")
                else:
                    # Fallback to older method if transformer not found
                    debug_print(f"Running victim with direct agent LD_PRELOAD={agent_path}")
                    env = os.environ.copy()
                    env["LD_PRELOAD"] = agent_path
                
                victim_cmd = [os.path.expanduser(VICTIM_PATH)]
                result = subprocess.run(victim_cmd, env=env, capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    debug_print(f"victim failed with exit code {result.returncode}")
                    debug_print(f"stdout: {result.stdout}")
                    debug_print(f"stderr: {result.stderr}")
                    continue
                
                avg_time = parse_victim_output(result.stdout)
                if avg_time:
                    results["userspace_syscall"].append(avg_time)
                    print(f"  Average time usage: {avg_time:.2f} ns")
                else:
                    debug_print(f"Failed to parse output: {result.stdout}")
            
            # Cleanup syscall server
            if 'syscall_proc' in locals() and syscall_proc:
                debug_print(f"Terminating syscall server process PID: {syscall_proc.pid}")
                syscall_proc.terminate()
                try:
                    syscall_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    debug_print("Syscall server process did not terminate, killing")
                    syscall_proc.kill()
                    
        except Exception as e:
            debug_print(f"Error in Method 2 (direct LD_PRELOAD): {e}")
            traceback.print_exc()
    
    except Exception as e:
        debug_print(f"Error in userspace BPF syscall test: {e}")
        traceback.print_exc()

def print_statistics():
    """Print benchmark statistics"""
    print("\n=== Benchmark Results ===")
    print(f"Each configuration run {NUM_RUNS} times")
    
    # Dictionary for formatted names for printing
    name_map = {
        "native": "Native (No tracing)",
        "kernel_tracepoint": "Kernel Tracepoint Syscall",
        "userspace_syscall": "Userspace BPF Syscall"
    }
    
    # For storing avg values for later comparison
    avgs = {}
    
    for test_name, values in results.items():
        if not values:
            print(f"\n{name_map.get(test_name, test_name)}: No valid results")
            continue
            
        avg = statistics.mean(values)
        avgs[test_name] = avg
        if len(values) > 1:
            median = statistics.median(values)
            stdev = statistics.stdev(values)
        else:
            median = values[0]
            stdev = 0
        min_val = min(values)
        max_val = max(values)
        
        print(f"\n{name_map.get(test_name, test_name)}:")
        print(f"  Average time usage (mean):   {avg:.2f} ns")
        print(f"  Average time usage (median): {median:.2f} ns")
        print(f"  Standard deviation:          {stdev:.2f}")
        print(f"  Min:                         {min_val:.2f} ns")
        print(f"  Max:                         {max_val:.2f} ns")
        print(f"  All runs:                    {[round(x, 2) for x in values]}")
    
    # Compare results
    if "native" in avgs:
        native_avg = avgs["native"]
        print("\n=== Overhead Compared to Native ===")
        
        for test_name, avg in avgs.items():
            if test_name != "native":
                impact = ((avg - native_avg) / native_avg) * 100
                print(f"{name_map.get(test_name, test_name)}: {impact:.2f}% increase")
        
    # Compare userBPF to kernel
    if "kernel_tracepoint" in avgs and "userspace_syscall" in avgs:
        kernel_avg = avgs["kernel_tracepoint"]
        userbpf_avg = avgs["userspace_syscall"]
        comparison = ((userbpf_avg - kernel_avg) / kernel_avg) * 100
        
        if comparison > 0:
            print(f"\nUserspace BPF syscall has {abs(comparison):.2f}% more overhead than kernel tracepoint")
        else:
            print(f"\nUserspace BPF syscall has {abs(comparison):.2f}% less overhead than kernel tracepoint")

def save_results():
    """Save results to a JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark/syscall/syscall_benchmark_results_{timestamp}.json"
    
    # Create a result object with detailed information
    result_obj = {
        "timestamp": timestamp,
        "runs": NUM_RUNS,
        "results": {},
        "system_info": {
            "os": " ".join(os.uname()),
            "hostname": os.uname().nodename,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Convert raw results to detailed statistics
    for test_name, values in results.items():
        if not values:
            result_obj["results"][test_name] = {"status": "no_data"}
            continue
            
        avg = statistics.mean(values)
        if len(values) > 1:
            median = statistics.median(values)
            stdev = statistics.stdev(values)
        else:
            median = values[0]
            stdev = 0
        
        result_obj["results"][test_name] = {
            "avg": avg,
            "median": median,
            "stdev": stdev,
            "min": min(values),
            "max": max(values),
            "raw_values": values
        }
    
    # Add comparison data if possible
    if "native" in result_obj["results"] and "native" in results and results["native"]:
        native_avg = statistics.mean(results["native"])
        result_obj["comparisons"] = {}
        
        for test_name, values in results.items():
            if test_name != "native" and values:
                avg = statistics.mean(values)
                overhead = ((avg - native_avg) / native_avg) * 100
                result_obj["comparisons"][f"{test_name}_vs_native"] = {
                    "overhead_percent": overhead
                }
        
        # Compare kernel vs userspace
        if "kernel_tracepoint" in results and results["kernel_tracepoint"] and \
           "userspace_syscall" in results and results["userspace_syscall"]:
            kernel_avg = statistics.mean(results["kernel_tracepoint"])
            userbpf_avg = statistics.mean(results["userspace_syscall"])
            difference = ((userbpf_avg - kernel_avg) / kernel_avg) * 100
            
            result_obj["comparisons"]["userspace_vs_kernel"] = {
                "difference_percent": difference
            }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(result_obj, f, indent=2)
    
    print(f"\nResults saved to {filename}")

def ensure_sudo():
    """Re-run the script with sudo if not already running with privileges"""
    if os.geteuid() != 0:
        print("This benchmark requires sudo privileges. Re-launching with sudo...")
        args = ['sudo', sys.executable] + sys.argv
        # Exit the current process and start a new one with sudo
        os.execvp('sudo', args)
        # If execvp fails, the script will continue below
        print("Failed to re-launch with sudo. Some tests may fail.")
        return False
    return True

def main():
    try:
        print("==== BPFtime Syscall Benchmark ====")
        
        # Check if we need to run with sudo
        if not ensure_sudo():
            print("WARNING: Some benchmarks require sudo privileges.")
            
        # For testing, reduce the number of runs
        global NUM_RUNS
        if "--test" in sys.argv:
            NUM_RUNS = 1
            debug_print("Running in test mode with NUM_RUNS=1")
            
        # Check for required files
        for path in [VICTIM_PATH, SYSCALL_BPF_PATH]:
            check_file_exists(os.path.expanduser(path))
            
        # Check for agent files
        check_file_exists(os.path.abspath(AGENT_PATH))
        check_file_exists(os.path.abspath(AGENT_TRANSFORMER_PATH))
        check_file_exists(os.path.abspath(SYSCALL_SERVER_PATH))
            
        # Check if bpftime exists but don't fail if it doesn't (we have fallback methods)
        check_file_exists(os.path.expanduser(BPFTIME_PATH))
            
        # Run benchmarks with cleanup between each
        try:
            cleanup_processes()
            run_native()
        except Exception as e:
            debug_print(f"Error in native benchmark: {e}")
            
        try:
            cleanup_processes()
            run_kernel_tracepoint()
        except Exception as e:
            debug_print(f"Error in kernel tracepoint benchmark: {e}")
            
        try:
            cleanup_processes()
            run_userspace_syscall()
        except Exception as e:
            debug_print(f"Error in userspace syscall benchmark: {e}")
            
        # Ensure final cleanup
        cleanup_processes()
        
        # Print and save results
        print_statistics()
        save_results()
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
    finally:
        debug_print("Cleaning up before exit")
        cleanup_processes()

if __name__ == "__main__":
    main() 