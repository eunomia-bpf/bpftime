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

# Configuration
NUM_RUNS = 10
VICTIM_PATH = "benchmark/syscall/victim"
SYSCALL_BPF_PATH = "benchmark/syscall/syscall"
AGENT_PATH = "build/runtime/agent/libbpftime-agent.so"
AGENT_TRANSFORMER_PATH = "build/attach/text_segment_transformer/libbpftime-agent-transformer.so"
SYSCALL_SERVER_PATH = "build/runtime/syscall-server/libbpftime-syscall-server.so"
BPFTIME_PATH = os.path.expanduser("~/.bpftime/bpftime")

# Result storage
results = {
    "native": [],               # No tracing
    "kernel_tracepoint": [],    # Kernel tracepoint syscall tracking
    "userspace_syscall": [],    # Userspace BPF syscall tracking
}

# Flag to prevent recursive cleanup
_cleanup_in_progress = False
_safe_pids = set()  # PIDs that should never be killed

# Initialize safe PIDs at startup
def init_safe_pids():
    """Initialize the list of PIDs that should never be killed"""
    global _safe_pids
    
    # Add our own PID and parent PID to safe list
    current_pid = os.getpid()
    _safe_pids.add(current_pid)
    
    try:
        parent_pid = os.getppid()
        _safe_pids.add(parent_pid)
    except:
        pass
    
    # Also add the shell's process group
    try:
        pgid = os.getpgid(0)
        _safe_pids.add(pgid)
    except:
        pass
    
    # Find all python processes with our script name to be extra safe
    script_name = os.path.basename(sys.argv[0])
    try:
        python_procs = subprocess.run(
            ["pgrep", "-f", f"python.*{script_name}"], 
            capture_output=True, 
            text=True
        ).stdout.strip()
        
        if python_procs:
            for pid in python_procs.split():
                try:
                    _safe_pids.add(int(pid))
                except:
                    pass
    except:
        pass
        
    debug_print(f"Initialized safe PIDs: {_safe_pids}")

def debug_print(message):
    """Print debug messages with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[DEBUG {timestamp}] {message}")

def cleanup_processes():
    """Kill any running victim or syscall processes"""
    global _cleanup_in_progress
    
    if _cleanup_in_progress:
        return
    
    try:
        _cleanup_in_progress = True
        debug_print("Cleaning up processes...")
        
        # Add current process and parent to safe PIDs
        current_pid = os.getpid()
        _safe_pids.add(current_pid)
        try:
            parent_pid = os.getppid()
            _safe_pids.add(parent_pid)
        except:
            pass
        
        debug_print(f"Safe PIDs: {_safe_pids}")
        
        # Define more specific patterns to avoid killing the benchmark script itself
        victim_basename = os.path.basename(VICTIM_PATH)
        syscall_basename = os.path.basename(SYSCALL_BPF_PATH)
        bpftime_basename = os.path.basename(BPFTIME_PATH)
        
        # Don't match our Python script
        script_name = os.path.basename(sys.argv[0])
        
        pgrep_cmds = [
            # Use exact name matching where possible
            ["pgrep", "-f", f"\\b{victim_basename}\\b"],
            ["pgrep", "-f", f"\\b{syscall_basename}\\b"],
            ["pgrep", "-f", f"\\b{bpftime_basename}\\b"]
        ]
        
        for cmd in pgrep_cmds:
            try:
                output = subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout.strip()
                if output:
                    pids = output.split()
                    for pid in pids:
                        try:
                            pid_int = int(pid)
                            # Double-check this isn't our own process or a parent
                            if pid_int not in _safe_pids:
                                # Additional safety check - verify this isn't the benchmark script
                                proc_cmd = subprocess.run(
                                    ["ps", "-p", pid, "-o", "cmd="], 
                                    capture_output=True, 
                                    text=True
                                ).stdout.strip()
                                
                                # Skip if this looks like our benchmark script
                                if script_name in proc_cmd and "python" in proc_cmd:
                                    debug_print(f"Skipping process {pid} (looks like benchmark script): {proc_cmd}")
                                    continue
                                    
                                debug_print(f"Killing process {pid}: {proc_cmd}")
                                subprocess.run(["sudo", "kill", "-9", pid], stderr=subprocess.DEVNULL, check=False)
                        except ValueError:
                            pass
                        except Exception as e:
                            debug_print(f"Error checking process {pid}: {e}")
            except Exception as e:
                debug_print(f"Error in pgrep command {cmd}: {e}")
                
        time.sleep(1)  # Give processes time to terminate
    except Exception as e:
        debug_print(f"Error during cleanup: {e}")
    finally:
        _cleanup_in_progress = False

def parse_victim_output(output):
    """Parse victim output to extract average time usage"""
    match = re.search(r'Average time usage\s+(\d+\.\d+)ns', output)
    if match:
        return float(match.group(1))
    debug_print("Failed to parse victim output")
    return None

def run_native():
    """Run baseline benchmarks (no tracing)"""
    print("\n=== Running Native Tests (No Tracing) ===")
    
    for i in range(NUM_RUNS):
        print(f"Run {i+1}/{NUM_RUNS}...")
        
        try:
            result = subprocess.run([VICTIM_PATH], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                debug_print(f"victim failed with exit code {result.returncode}")
                continue
            
            avg_time = parse_victim_output(result.stdout)
            if avg_time:
                results["native"].append(avg_time)
                print(f"  Average time usage: {avg_time:.2f} ns")
        except Exception as e:
            debug_print(f"Error in native run: {e}")
            cleanup_processes()

def run_kernel_tracepoint():
    """Run kernel tracepoint benchmarks"""
    print("\n=== Running Kernel Tracepoint Tests ===")
    
    syscall_proc = None
    
    try:
        # Start syscall BPF program
        syscall_proc = subprocess.Popen(
            [SYSCALL_BPF_PATH], 
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        _safe_pids.add(syscall_proc.pid)
        
        # Give time to start
        time.sleep(2)
        
        for i in range(NUM_RUNS):
            print(f"Run {i+1}/{NUM_RUNS}...")
            
            try:
                result = subprocess.run([VICTIM_PATH], capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    debug_print(f"victim failed with exit code {result.returncode}")
                    continue
                
                avg_time = parse_victim_output(result.stdout)
                if avg_time:
                    results["kernel_tracepoint"].append(avg_time)
                    print(f"  Average time usage: {avg_time:.2f} ns")
            except Exception as e:
                debug_print(f"Error in kernel tracepoint run: {e}")
                cleanup_processes()
    finally:
        if syscall_proc and syscall_proc.poll() is None:
            syscall_proc.terminate()
            try:
                syscall_proc.wait(timeout=5)
            except:
                syscall_proc.kill()
        cleanup_processes()

def run_userspace_syscall():
    """Run userspace BPF syscall benchmarks"""
    print("\n=== Running Userspace BPF Syscall Tests ===")
    
    syscall_proc = None
    
    try:
        # Check if bpftime exists and try that method first
        if os.path.exists(BPFTIME_PATH):
            debug_print("Using bpftime command method")
            # Load the syscall BPF program
            try:
                load_result = subprocess.run(
                    [BPFTIME_PATH, "load", SYSCALL_BPF_PATH], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                if load_result.returncode != 0:
                    debug_print(f"bpftime load failed: {load_result.stderr}")
                    debug_print("Falling back to LD_PRELOAD method")
                else:
                    debug_print("bpftime loaded successfully")
                    time.sleep(2)  # Give time to initialize
                    
                    for i in range(NUM_RUNS):
                        print(f"Run {i+1}/{NUM_RUNS}...")
                        
                        try:
                            result = subprocess.run(
                                [BPFTIME_PATH, "start", "-s", VICTIM_PATH], 
                                capture_output=True, 
                                text=True, 
                                timeout=60
                            )
                            
                            if result.returncode != 0:
                                debug_print(f"bpftime start failed with code {result.returncode}")
                                debug_print(f"Error: {result.stderr}")
                                continue
                            
                            avg_time = parse_victim_output(result.stdout)
                            if avg_time:
                                results["userspace_syscall"].append(avg_time)
                                print(f"  Average time usage: {avg_time:.2f} ns")
                        except Exception as e:
                            debug_print(f"Error in bpftime method: {e}")
                            cleanup_processes()
                    
                    # Unload bpftime
                    try:
                        unload_result = subprocess.run(
                            [BPFTIME_PATH, "unload"], 
                            capture_output=True,
                            timeout=10
                        )
                        if unload_result.returncode != 0:
                            debug_print(f"bpftime unload failed: {unload_result.stderr}")
                            cleanup_processes()
                    except Exception as e:
                        debug_print(f"Error unloading bpftime: {e}")
                        cleanup_processes()
                    
                    # If we got results, return
                    if results["userspace_syscall"]:
                        return
            except Exception as e:
                debug_print(f"Error using bpftime method: {e}")
                # Cleanup any lingering processes before trying next method
                cleanup_processes()
                
        # Fallback to LD_PRELOAD method if bpftime failed or doesn't exist
        debug_print("Using LD_PRELOAD method")
        
        # Check if required files exist
        if not os.path.exists(SYSCALL_SERVER_PATH):
            debug_print(f"Missing required file: {SYSCALL_SERVER_PATH}")
            return
        
        if not os.path.exists(AGENT_PATH) or not os.path.exists(AGENT_TRANSFORMER_PATH):
            debug_print(f"Missing required files for agent: {AGENT_PATH} or {AGENT_TRANSFORMER_PATH}")
            return
            
        # Start syscall server with LD_PRELOAD
        env = os.environ.copy()
        env["LD_PRELOAD"] = SYSCALL_SERVER_PATH
        
        try:
            syscall_proc = subprocess.Popen(
                [SYSCALL_BPF_PATH],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            _safe_pids.add(syscall_proc.pid)
            
            # Check if process is still running after a short delay
            time.sleep(1)
            if syscall_proc.poll() is not None:
                debug_print(f"Syscall server exited immediately with code {syscall_proc.returncode}")
                return
                
            debug_print(f"Syscall server started with PID: {syscall_proc.pid}")
            
            # Wait for server to initialize
            time.sleep(2)
            
            for i in range(NUM_RUNS):
                print(f"Run {i+1}/{NUM_RUNS}...")
                
                # Run victim with agent LD_PRELOAD
                env = os.environ.copy()
                env["AGENT_SO"] = AGENT_PATH
                env["LD_PRELOAD"] = AGENT_TRANSFORMER_PATH
                
                try:
                    result = subprocess.run(
                        [VICTIM_PATH], 
                        env=env, 
                        capture_output=True, 
                        text=True, 
                        timeout=60
                    )
                    
                    if result.returncode != 0:
                        debug_print(f"victim failed with exit code {result.returncode}")
                        debug_print(f"Error: {result.stderr}")
                        continue
                    
                    avg_time = parse_victim_output(result.stdout)
                    if avg_time:
                        results["userspace_syscall"].append(avg_time)
                        print(f"  Average time usage: {avg_time:.2f} ns")
                    else:
                        debug_print(f"Failed to parse output: {result.stdout}")
                except Exception as e:
                    debug_print(f"Error in LD_PRELOAD method: {e}")
                    cleanup_processes()
        except Exception as e:
            debug_print(f"Error starting syscall server: {e}")
    finally:
        if syscall_proc and syscall_proc.poll() is None:
            debug_print(f"Terminating syscall server (PID: {syscall_proc.pid})")
            try:
                syscall_proc.terminate()
                try:
                    syscall_proc.wait(timeout=5)
                except:
                    debug_print("Syscall server didn't terminate, force killing")
                    syscall_proc.kill()
            except Exception as e:
                debug_print(f"Error terminating syscall server: {e}")
        cleanup_processes()

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
    
    try:
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
    except Exception as e:
        debug_print(f"Error saving results: {e}")

def save_markdown_results():
    """Save results to a human-readable Markdown file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = "benchmark/syscall/results.md"
    
    try:
        # Dictionary for formatted names for printing
        name_map = {
            "native": "Native (No tracing)",
            "kernel_tracepoint": "Kernel Tracepoint Syscall",
            "userspace_syscall": "Userspace BPF Syscall"
        }
        
        # Prepare markdown content
        markdown = []
        markdown.append("# BPFtime Syscall Benchmark Results")
        markdown.append(f"\nBenchmark run at: **{timestamp}**\n")
        
        # System information
        markdown.append("## System Information")
        markdown.append(f"- OS: {' '.join(os.uname())}")
        markdown.append(f"- Hostname: {os.uname().nodename}")
        markdown.append(f"- Number of runs per configuration: {NUM_RUNS}\n")
        
        # Results for each configuration
        markdown.append("## Benchmark Results")
        
        for test_name, values in results.items():
            if not values:
                markdown.append(f"\n### {name_map.get(test_name, test_name)}: No valid results")
                continue
                
            markdown.append(f"\n### {name_map.get(test_name, test_name)}")
            
            avg = statistics.mean(values)
            if len(values) > 1:
                median = statistics.median(values)
                stdev = statistics.stdev(values)
            else:
                median = values[0]
                stdev = 0
            min_val = min(values)
            max_val = max(values)
            
            markdown.append("| Metric | Value |")
            markdown.append("| ------ | ----- |")
            markdown.append(f"| Average time usage (mean) | **{avg:.2f} ns** |")
            markdown.append(f"| Average time usage (median) | {median:.2f} ns |")
            markdown.append(f"| Standard deviation | {stdev:.2f} |")
            markdown.append(f"| Min | {min_val:.2f} ns |")
            markdown.append(f"| Max | {max_val:.2f} ns |")
            
            markdown.append("\n**Individual runs (ns):**")
            runs_formatted = ', '.join([f"{round(x, 2)}" for x in values])
            markdown.append(f"`{runs_formatted}`\n")
        
        # Comparison results
        markdown.append("## Comparison Results")
        
        # Compare to native
        avgs = {}
        for test_name, values in results.items():
            if values:
                avgs[test_name] = statistics.mean(values)
                
        if "native" in avgs:
            native_avg = avgs["native"]
            markdown.append("\n### Overhead Compared to Native")
            
            markdown.append("| Configuration | Overhead |")
            markdown.append("| ------------ | -------- |")
            
            for test_name, avg in avgs.items():
                if test_name != "native":
                    impact = ((avg - native_avg) / native_avg) * 100
                    markdown.append(f"| {name_map.get(test_name, test_name)} | **{impact:.2f}%** |")
        
        # Compare userBPF to kernel
        if "kernel_tracepoint" in avgs and "userspace_syscall" in avgs:
            markdown.append("\n### Userspace BPF vs Kernel Tracepoint")
            
            kernel_avg = avgs["kernel_tracepoint"]
            userbpf_avg = avgs["userspace_syscall"]
            comparison = ((userbpf_avg - kernel_avg) / kernel_avg) * 100
            
            if comparison > 0:
                markdown.append(f"Userspace BPF syscall has **{abs(comparison):.2f}%** more overhead than kernel tracepoint")
            else:
                markdown.append(f"Userspace BPF syscall has **{abs(comparison):.2f}%** less overhead than kernel tracepoint")
        
        # Add summary and conclusions
        markdown.append("\n## Summary")
        markdown.append("This benchmark compares three configurations for syscall handling:")
        markdown.append("1. **Native**: No tracing or interception")
        markdown.append("2. **Kernel Tracepoint**: Traditional kernel-based syscall tracking")
        markdown.append("3. **Userspace BPF**: BPFtime's userspace syscall interception")
        
        markdown.append("\nEach configuration was run multiple times to ensure statistical significance. Lower numbers represent better performance.")
        
        # Save to file
        with open(filename, 'w') as f:
            f.write('\n'.join(markdown))
        
        print(f"\nHuman-readable results saved to {filename}")
    except Exception as e:
        debug_print(f"Error saving markdown results: {e}")
        traceback.print_exc()

def main():
    print("==== BPFtime Syscall Benchmark ====")
    
    # Initialize safe PIDs to prevent the script from killing itself
    init_safe_pids()
    
    # Check for sudo
    if os.geteuid() != 0:
        print("This benchmark requires sudo privileges. Please run with sudo.")
        return
    
    # Initial cleanup to ensure no leftover processes
    cleanup_processes()
    
    # Run benchmarks
    try:
        run_native()
        cleanup_processes()
        
        run_kernel_tracepoint()
        cleanup_processes()
        
        run_userspace_syscall()
        cleanup_processes()
        
        # Print and save results
        print_statistics()
        save_results()
        save_markdown_results()
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
    finally:
        cleanup_processes()
        print("Benchmark complete")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error in main: {e}")
        traceback.print_exc()
        # Final attempt to clean up
        try:
            cleanup_processes()
        except:
            pass 