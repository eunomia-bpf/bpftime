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
WRK_CMD = ["wrk", "http://127.0.0.1:801/index.html", "-c", "100", "-d", "10"]
NGINX_CMD = ["nginx", "-c", "nginx.conf", "-p", "benchmark/syscount-nginx"]
TEST_URL = "http://127.0.0.1:801/index.html"
SYSCOUNT_PATH = "example/libbpf-tools/syscount/syscount"
AGENT_PATH = "build/runtime/agent/libbpftime-agent.so"
SYSCALL_SERVER_PATH = "build/runtime/syscall-server/libbpftime-syscall-server.so"

# Result storage
results = {
    "native": [],               # No tracing
    "kernel_targeted": [],      # Kernel syscount targeting nginx pid
    "kernel_untargeted": [],    # Kernel syscount not targeting nginx
    "userbpf_targeted": [],     # Userspace syscount targeting nginx pid
    "userbpf_untargeted": [],   # Userspace syscount not targeting nginx
}

# Define a signal handler to prevent unexpected termination
def signal_handler(sig, frame):
    debug_print(f"Received signal {sig}, gracefully exiting...")
    sys.exit(0)

# Register the signal handler for common signals
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def debug_print(message):
    """Print debug messages with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[DEBUG {timestamp}] {message}")

def remove_access_log():
    """Remove the nginx access log file"""
    log_path = os.path.join("benchmark", "syscount-nginx", "access.log")
    abs_log_path = os.path.abspath(log_path)
    
    debug_print(f"Removing access log: {abs_log_path}")
    try:
        if os.path.exists(abs_log_path):
            os.remove(abs_log_path)
            debug_print("Access log removed successfully")
        else:
            debug_print("Access log not found (this might be normal on first run)")
    except Exception as e:
        debug_print(f"Error removing access log: {e}")

def check_file_exists(path):
    """Check if a file exists and print its absolute path"""
    abs_path = os.path.abspath(path)
    exists = os.path.exists(abs_path)
    debug_print(f"Checking file: {abs_path} - {'EXISTS' if exists else 'NOT FOUND'}")
    return exists

def check_command_exists(cmd):
    """Check if a command exists in PATH"""
    try:
        result = subprocess.run(["which", cmd], capture_output=True, text=True)
        exists = result.returncode == 0
        debug_print(f"Checking command: {cmd} - {'EXISTS' if exists else 'NOT FOUND'}")
        
        if exists:
            debug_print(f"  Path: {result.stdout.strip()}")
        else:
            # For nginx, check common locations
            if cmd == "nginx":
                common_paths = [
                    "/usr/sbin/nginx", 
                    "/usr/local/sbin/nginx",
                    "/usr/local/bin/nginx",
                    "/opt/nginx/sbin/nginx"
                ]
                for path in common_paths:
                    if os.path.exists(path):
                        debug_print(f"Found {cmd} at {path}")
                        # Update global nginx command
                        global NGINX_CMD
                        NGINX_CMD[0] = path
                        return True
                
                debug_print("Could not find nginx in common locations")
        
        return exists
    except Exception as e:
        debug_print(f"Error checking command {cmd}: {e}")
        return False

def cleanup_processes():
    """Kill any running nginx or syscount processes"""
    try:
        debug_print("Cleaning up processes...")
        
        # Get the PIDs of nginx and syscount processes
        nginx_pids = subprocess.run(["pgrep", "-x", "nginx"], capture_output=True, text=True).stdout.strip().split()
        syscount_pids = subprocess.run(["pgrep", "-x", "syscount"], capture_output=True, text=True).stdout.strip().split()
        
        debug_print(f"Found nginx PIDs: {nginx_pids}")
        debug_print(f"Found syscount PIDs: {syscount_pids}")
        
        # Kill nginx processes by PID
        for pid in nginx_pids:
            try:
                debug_print(f"Terminating nginx process with PID {pid}")
                subprocess.run(["kill", pid], stderr=subprocess.DEVNULL, check=False)
            except Exception as e:
                debug_print(f"Error terminating nginx PID {pid}: {e}")
        
        # Kill syscount processes by PID
        for pid in syscount_pids:
            try:
                debug_print(f"Terminating syscount process with PID {pid}")
                subprocess.run(["kill", pid], stderr=subprocess.DEVNULL, check=False)
            except Exception as e:
                debug_print(f"Error terminating syscount PID {pid}: {e}")
                try:
                    debug_print(f"Trying with sudo...")
                    subprocess.run(["sudo", "kill", pid], stderr=subprocess.DEVNULL, check=False)
                except Exception as e2:
                    debug_print(f"Error with sudo: {e2}")
        
        # Wait for processes to terminate
        time.sleep(1)
        
        # Check if any processes are still running and try forceful termination if needed
        remaining_nginx_pids = subprocess.run(["pgrep", "-x", "nginx"], capture_output=True, text=True).stdout.strip().split()
        remaining_syscount_pids = subprocess.run(["pgrep", "-x", "syscount"], capture_output=True, text=True).stdout.strip().split()
        
        debug_print(f"After cleanup: nginx PIDs: {remaining_nginx_pids}, syscount PIDs: {remaining_syscount_pids}")
        
        # Force kill remaining processes
        for pid in remaining_nginx_pids:
            try:
                debug_print(f"Force killing nginx process with PID {pid}")
                subprocess.run(["kill", "-9", pid], stderr=subprocess.DEVNULL, check=False)
            except Exception as e:
                debug_print(f"Error force killing nginx PID {pid}: {e}")
        
        for pid in remaining_syscount_pids:
            try:
                debug_print(f"Force killing syscount process with PID {pid}")
                subprocess.run(["kill", "-9", pid], stderr=subprocess.DEVNULL, check=False)
            except Exception as e:
                debug_print(f"Error force killing syscount PID {pid}: {e}")
                try:
                    debug_print(f"Trying with sudo...")
                    subprocess.run(["sudo", "kill", "-9", pid], stderr=subprocess.DEVNULL, check=False)
                except Exception as e2:
                    debug_print(f"Error with sudo: {e2}")
    
    except Exception as e:
        debug_print(f"Error during cleanup: {e}")
        traceback.print_exc()

def parse_wrk_output(output):
    """Parse wrk output to extract requests/sec"""
    debug_print(f"Parsing wrk output: {output[:100]}...")  # Print first 100 chars
    match = re.search(r'Requests/sec:\s+(\d+\.\d+)', output)
    if match:
        return float(match.group(1))
    debug_print("Failed to parse wrk output")
    return None

def start_nginx():
    """Start nginx server and return process and PID"""
    cleanup_processes()
    remove_access_log()  # Remove access log before starting nginx
    
    # Check if nginx exists
    check_command_exists("nginx")
    
    # Check current directory
    debug_print(f"Current directory: {os.getcwd()}")
    nginx_conf = "benchmark/syscount-nginx/nginx.conf"
    debug_print(f"Checking if nginx.conf exists: {os.path.exists(nginx_conf)}")
    
    if not os.path.exists(nginx_conf):
        debug_print(f"ERROR: {nginx_conf} not found!")
        return None, None
    
    # Start nginx with full path
    abs_nginx_conf = os.path.abspath(nginx_conf)
    abs_nginx_dir = os.path.dirname(abs_nginx_conf)
    modified_nginx_cmd = ["nginx", "-c", abs_nginx_conf, "-p", abs_nginx_dir]
    debug_print(f"Starting nginx with command: {' '.join(modified_nginx_cmd)}")
    
    try:
        nginx_proc = subprocess.Popen(modified_nginx_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)  # Give nginx time to start
        
        # Check if nginx started successfully
        if nginx_proc.poll() is not None:
            stdout, stderr = nginx_proc.communicate()
            debug_print(f"nginx failed to start. Exit code: {nginx_proc.returncode}")
            debug_print(f"stdout: {stdout.decode() if stdout else 'None'}")
            debug_print(f"stderr: {stderr.decode() if stderr else 'None'}")
            return None, None
        
        debug_print("nginx started successfully")
        
        # Check if nginx is listening
        try:
            curl_check = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", TEST_URL], 
                                        capture_output=True, text=True, timeout=5)
            debug_print(f"HTTP status code from nginx: {curl_check.stdout}")
            
            # Get nginx PID
            nginx_pid = subprocess.run(["pgrep", "-f", "nginx -c"], capture_output=True, text=True).stdout.strip()
            debug_print(f"nginx PID: {nginx_pid}")
            return nginx_proc, nginx_pid
            
        except Exception as e:
            debug_print(f"Error checking nginx: {e}")
            return nginx_proc, None
    
    except Exception as e:
        debug_print(f"Error starting nginx: {e}")
        traceback.print_exc()
        return None, None

def run_native():
    """Run baseline benchmarks (no tracing)"""
    print("\n=== Running Native Tests (No Tracing) ===")
    
    nginx_proc, nginx_pid = start_nginx()
    if not nginx_proc:
        debug_print("Failed to start nginx, skipping native test")
        return
    
    try:
        for i in range(NUM_RUNS):
            print(f"Run {i+1}/{NUM_RUNS}...")
            debug_print(f"Running wrk with command: {' '.join(WRK_CMD)}")
            
            # Run wrk
            try:
                result = subprocess.run(WRK_CMD, capture_output=True, text=True, timeout=15)
                debug_print(f"wrk exit code: {result.returncode}")
                
                if result.returncode != 0:
                    debug_print(f"wrk failed with exit code {result.returncode}")
                    debug_print(f"stdout: {result.stdout}")
                    debug_print(f"stderr: {result.stderr}")
                    continue
                
                req_per_sec = parse_wrk_output(result.stdout)
                if req_per_sec:
                    results["native"].append(req_per_sec)
                    print(f"  Requests/sec: {req_per_sec:.2f}")
                else:
                    debug_print(f"Failed to parse output: {result.stdout}")
            except subprocess.TimeoutExpired:
                debug_print("wrk command timed out")
            except Exception as e:
                debug_print(f"Error running wrk: {e}")
                traceback.print_exc()
    
    except Exception as e:
        debug_print(f"Error in native test: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        if nginx_proc:
            debug_print("Terminating nginx")
            try:
                nginx_proc.terminate()
                nginx_proc.wait(timeout=5)
            except Exception as e:
                debug_print(f"Error terminating nginx: {e}")
                try:
                    nginx_proc.kill()
                except:
                    pass

def run_kernel_syscount(target_pid=None):
    """
    Run kernel syscount benchmarks
    If target_pid is provided, syscount will target that PID
    Otherwise, it will track all processes
    """
    test_name = "kernel_targeted" if target_pid else "kernel_untargeted"
    print(f"\n=== Running Kernel syscount Tests ({test_name}) ===")
    
    # Start nginx
    nginx_proc, nginx_pid = start_nginx()
    if not nginx_proc:
        debug_print("Failed to start nginx, skipping kernel syscount test")
        return
    
    # Check if syscount exists
    if not check_file_exists(SYSCOUNT_PATH):
        debug_print(f"Skipping kernel syscount tests: {SYSCOUNT_PATH} not found")
        return
    
    try:
        for i in range(NUM_RUNS):
            print(f"Run {i+1}/{NUM_RUNS}...")

            if not nginx_pid:
                debug_print("nginx PID not found, skipping syscount")
                continue
            
            # Start syscount
            syscount_cmd = ["sudo", SYSCOUNT_PATH, "-d", "20"]
            if target_pid:
                syscount_cmd.extend(["-p", nginx_pid])
            else:
                # target another random pid
                syscount_cmd.extend(["-p", "1"])
            
            debug_print(f"Starting kernel syscount: {' '.join(syscount_cmd)}")
            try:
                syscount_proc = subprocess.Popen(
                    syscount_cmd, 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                time.sleep(2)  # Give syscount time to start
                
                # Run wrk
                debug_print(f"Running wrk with command: {' '.join(WRK_CMD)}")
                result = subprocess.run(WRK_CMD, capture_output=True, text=True, timeout=15)
                req_per_sec = parse_wrk_output(result.stdout)
                
                if req_per_sec:
                    results[test_name].append(req_per_sec)
                    print(f"  Requests/sec: {req_per_sec:.2f}")
                else:
                    debug_print(f"Failed to parse output: {result.stdout}")
                
                # Wait for syscount to finish
                try:
                    debug_print("Waiting for syscount to finish...")
                    syscount_proc.wait(timeout=25)  # Give 5 seconds more than duration
                except subprocess.TimeoutExpired:
                    debug_print("syscount didn't finish within expected time, terminating...")
                    subprocess.run(["sudo", "pkill", "-f", "syscount"], stderr=subprocess.DEVNULL)
                    
            except Exception as e:
                debug_print(f"Error in kernel syscount run: {e}")
                traceback.print_exc()
    
    except Exception as e:
        debug_print(f"Error in kernel syscount test: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        debug_print("Terminating nginx")
        try:
            if nginx_proc:
                nginx_proc.terminate()
                nginx_proc.wait(timeout=5)
        except Exception as e:
            debug_print(f"Error terminating nginx: {e}")
            try:
                nginx_proc.kill()
            except:
                pass

def run_userbpf_syscount(target_pid=None):
    """
    Run userspace BPF syscount benchmarks
    If target_pid is provided, syscount will target that PID
    Otherwise, it will track all processes
    """
    test_name = "userbpf_targeted" if target_pid else "userbpf_untargeted"
    print(f"\n=== Running UserBPF syscount Tests ({test_name}) ===")
    
    # Check if required files exist
    if not check_file_exists(SYSCOUNT_PATH) or not check_file_exists(AGENT_PATH) or not check_file_exists(SYSCALL_SERVER_PATH):
        debug_print("Skipping userspace BPF syscount tests: required files not found")
        return
    
    for i in range(NUM_RUNS):
        print(f"Run {i+1}/{NUM_RUNS}...")
        
        try:
            # Start nginx with bpftime
            debug_print("Starting nginx with bpftime")
            env = os.environ.copy()
            env["LD_PRELOAD"] = AGENT_PATH
            
            # Use the same nginx path approach as baseline
            nginx_conf = "benchmark/syscount-nginx/nginx.conf"
            if not os.path.exists(nginx_conf):
                debug_print(f"ERROR: {nginx_conf} not found!")
                continue
            
            # Start nginx with full path
            abs_nginx_conf = os.path.abspath(nginx_conf)
            abs_nginx_dir = os.path.dirname(abs_nginx_conf)
            modified_nginx_cmd = ["nginx", "-c", abs_nginx_conf, "-p", abs_nginx_dir]
            
            debug_print(f"Starting nginx with bpftime: {' '.join(modified_nginx_cmd)}")
            nginx_proc = subprocess.Popen(modified_nginx_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(2)  # Give nginx time to start
            
            # Check if nginx started successfully
            if nginx_proc.poll() is not None:
                stdout, stderr = nginx_proc.communicate()
                debug_print(f"nginx with bpftime failed to start. Exit code: {nginx_proc.returncode}")
                debug_print(f"stdout: {stdout.decode() if stdout else 'None'}")
                debug_print(f"stderr: {stderr.decode() if stderr else 'None'}")
                continue
            
            # Get nginx PID
            nginx_pid = subprocess.run(["pgrep", "-f", "nginx -c"], capture_output=True, text=True).stdout.strip()
            debug_print(f"nginx PID: {nginx_pid}")
            if not nginx_pid:
                debug_print("nginx PID not found, skipping syscount")
                continue
            
            # Start syscount with bpftime
            syscount_cmd = [SYSCOUNT_PATH, "-d", "20"]
            if target_pid:
                syscount_cmd.extend(["-p", nginx_pid])
            elif not target_pid:
                # target another random pid
                syscount_cmd.extend(["-p", "1"])
            debug_print(f"Starting syscount with bpftime: {' '.join(syscount_cmd)}")
            env = os.environ.copy()
            env["LD_PRELOAD"] = SYSCALL_SERVER_PATH
            
            syscount_proc = subprocess.Popen(
                syscount_cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(2)  # Give syscount time to start
            
            # Run wrk
            debug_print(f"Running wrk with command: {' '.join(WRK_CMD)}")
            result = subprocess.run(WRK_CMD, capture_output=True, text=True, timeout=15)
            req_per_sec = parse_wrk_output(result.stdout)
            
            if req_per_sec:
                results[test_name].append(req_per_sec)
                print(f"  Requests/sec: {req_per_sec:.2f}")
            else:
                debug_print(f"Failed to parse output: {result.stdout}")
            
            # Wait for syscount to finish
            try:
                debug_print("Waiting for syscount to finish...")
                syscount_proc.wait(timeout=25)  # Give 5 seconds more than duration
            except subprocess.TimeoutExpired:
                debug_print("syscount didn't finish within expected time, terminating...")
                syscount_proc.terminate()
                
        except Exception as e:
            debug_print(f"Error in userspace BPF syscount run: {e}")
            traceback.print_exc()
        finally:
            # Cleanup
            debug_print("Terminating processes")
            try:
                if 'nginx_proc' in locals():
                    nginx_proc.terminate()
                    nginx_proc.wait(timeout=5)
                if 'syscount_proc' in locals():
                    syscount_proc.terminate()
                    syscount_proc.wait(timeout=5)
            except Exception as e:
                debug_print(f"Error terminating processes: {e}")
            
            cleanup_processes()

def print_statistics():
    """Print benchmark statistics"""
    print("\n=== Benchmark Results ===")
    print(f"Each configuration run {NUM_RUNS} times")
    
    # Dictionary for formatted names for printing
    name_map = {
        "native": "Native (No tracing)",
        "kernel_targeted": "Kernel syscount (targeting nginx)",
        "kernel_untargeted": "Kernel syscount (not targeting nginx)",
        "userbpf_targeted": "UserBPF syscount (targeting nginx)",
        "userbpf_untargeted": "UserBPF syscount (not targeting nginx)"
    }
    
    # For storing avg values for later comparison
    avgs = {}
    
    for test_name, values in results.items():
        if not values:
            print(f"\n{name_map.get(test_name, test_name)}: No valid results")
            continue
            
        avg = statistics.mean(values)
        avgs[test_name] = avg
        median = statistics.median(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        min_val = min(values)
        max_val = max(values)
        
        print(f"\n{name_map.get(test_name, test_name)}:")
        print(f"  Requests/sec (mean):   {avg:.2f}")
        print(f"  Requests/sec (median): {median:.2f}")
        print(f"  Standard deviation:    {stdev:.2f}")
        print(f"  Min:                   {min_val:.2f}")
        print(f"  Max:                   {max_val:.2f}")
        print(f"  All runs:              {[round(x, 2) for x in values]}")
    
    # Compare results
    if "native" in avgs:
        native_avg = avgs["native"]
        print("\n=== Performance Impact Compared to Native ===")
        
        for test_name, avg in avgs.items():
            if test_name != "native":
                impact = ((native_avg - avg) / native_avg) * 100
                print(f"{name_map.get(test_name, test_name)}: {impact:.2f}% decrease")
        
    # Compare userBPF to kernel
    if "kernel_targeted" in avgs and "userbpf_targeted" in avgs:
        kernel_avg = avgs["kernel_targeted"]
        userbpf_avg = avgs["userbpf_targeted"]
        improvement = ((userbpf_avg - kernel_avg) / kernel_avg) * 100
        print(f"\nUserBPF improvement over kernel (targeted): {improvement:.2f}%")
    
    if "kernel_untargeted" in avgs and "userbpf_untargeted" in avgs:
        kernel_avg = avgs["kernel_untargeted"]
        userbpf_avg = avgs["userbpf_untargeted"]
        improvement = ((userbpf_avg - kernel_avg) / kernel_avg) * 100
        print(f"UserBPF improvement over kernel (untargeted): {improvement:.2f}%")

def save_results():
    """Save results to a JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark/syscount-nginx/benchmark_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "runs": NUM_RUNS,
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to {filename}")

def main():
    try:
        # For testing, reduce the number of runs
        global NUM_RUNS
        if "--test" in sys.argv:
            NUM_RUNS = 1
            debug_print("Running in test mode with NUM_RUNS=1")
        
        debug_print("Starting benchmark script")
        debug_print(f"Current working directory: {os.getcwd()}")
        
        # Check if nginx is installed
        nginx_installed = check_command_exists("nginx")
        if not nginx_installed:
            debug_print("ERROR: nginx command not found!")
            debug_print("Please install nginx before running this benchmark.")
            debug_print("On Ubuntu/Debian: sudo apt-get install nginx")
            debug_print("On CentOS/RHEL: sudo yum install nginx")
            return
        
        # Check if we're in the right directory
        if not os.path.exists("benchmark/syscount-nginx/nginx.conf"):
            debug_print("nginx.conf not found in expected location")
            debug_print("Checking if we need to adjust paths...")
            
            # Try to find the correct directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            debug_print(f"Script directory: {script_dir}")
            
            # If we're running from the benchmark/syscount-nginx directory, adjust paths
            if os.path.basename(script_dir) == "syscount-nginx":
                debug_print("Running from syscount-nginx directory, adjusting paths")
                os.chdir(os.path.dirname(os.path.dirname(script_dir)))  # Go up two levels
                debug_print(f"New working directory: {os.getcwd()}")
        
        # Check if we can access necessary files
        for path in [AGENT_PATH, SYSCALL_SERVER_PATH, SYSCOUNT_PATH]:
            check_file_exists(path)
        
        # Check if nginx.conf exists
        nginx_conf_path = "benchmark/syscount-nginx/nginx.conf"
        if not os.path.exists(nginx_conf_path):
            debug_print(f"ERROR: {nginx_conf_path} not found!")
            debug_print("This is required for the benchmark to run.")
            return
        
        # Check if wrk is available
        if not check_command_exists("wrk"):
            debug_print("ERROR: wrk command not found!")
            debug_print("Please install wrk before running this benchmark.")
            return
        
        # Run benchmarks
        run_native()                      # No tracing
        run_kernel_syscount(target_pid=True)  # Kernel syscount targeting nginx
        run_kernel_syscount(target_pid=False) # Kernel syscount not targeting nginx
        run_userbpf_syscount(target_pid=True) # UserBPF syscount targeting nginx
        run_userbpf_syscount(target_pid=False)# UserBPF syscount not targeting nginx
        
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