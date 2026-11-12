#!/usr/bin/env python3
"""
Utility functions for the nginx benchmark script
"""

import os
import subprocess
import time
import datetime
import select
import shutil
import sys
from pathlib import Path

# Benchmark log file
BENCHMARK_LOG = None

def setup_log(log_path):
    """Set up the log file path"""
    global BENCHMARK_LOG
    BENCHMARK_LOG = log_path

def log_message(message, also_print=True):
    """Log a message to the benchmark log file and optionally print to console"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(BENCHMARK_LOG, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")
    if also_print:
        print(message)

def check_prerequisites(required_tools, nginx_bin):
    """Check if all required tools are available"""
    log_message("\n=== Checking prerequisites ===")
    
    for tool in required_tools:
        if shutil.which(tool) is None:
            log_message(f"Error: {tool} is not installed or not in PATH")
            log_message(f"Please install {tool} and try again")
            sys.exit(1)
    
    if not os.path.exists(nginx_bin):
        log_message(f"Error: Nginx binary not found at {nginx_bin}")
        log_message("Please build the project first")
        sys.exit(1)
    
    log_message("All prerequisites are met")

def start_nginx(nginx_bin, config_path, working_dir, env=None):
    """Start nginx with the specified configuration"""
    cmd = [nginx_bin, "-p", str(working_dir), "-c", config_path]
    cmd_str = ' '.join(cmd)
    log_message(f"Starting nginx with command: {cmd_str}")
    
    # If environment variables are provided, log them
    if env:
        for key, value in env.items():
            log_message(f"Environment variable: {key}={value}")
    
    # Start nginx as a subprocess - using binary mode is safer for reading output
    process = subprocess.Popen(cmd, 
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              cwd=working_dir,
                              env=env)
    
    # Give nginx time to start
    time.sleep(2)
    
    # Check if nginx is running
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        error_msg = f"Error starting nginx: {stderr.decode('utf-8', errors='replace')}"
        log_message(error_msg)
        log_message(f"Nginx stdout: {stdout.decode('utf-8', errors='replace')}")
        log_message(f"Nginx stderr: {stderr.decode('utf-8', errors='replace')}")
        return None
    
    return process

def collect_process_output(process, prefix):
    """Non-blocking collection of process output"""
    if process and process.poll() is None:
        # Get output without blocking
        stdout_data = b""
        stderr_data = b""
        
        # Try to read stdout if available and it's a pipe
        if process.stdout and hasattr(process.stdout, 'fileno'):
            try:
                # Check if there's data to read without blocking
                r, _, _ = select.select([process.stdout], [], [], 0)
                if r:
                    stdout_data = os.read(process.stdout.fileno(), 1024)
            except (ValueError, IOError, OSError):
                pass
        
        # Try to read stderr if available and it's a pipe
        if process.stderr and hasattr(process.stderr, 'fileno'):
            try:
                # Check if there's data to read without blocking
                r, _, _ = select.select([process.stderr], [], [], 0)
                if r:
                    stderr_data = os.read(process.stderr.fileno(), 1024)
            except (ValueError, IOError, OSError):
                pass
            
        # Log any output we got
        if stdout_data:
            log_message(f"{prefix} stdout: {stdout_data.decode('utf-8', errors='replace')}", also_print=False)
        if stderr_data:
            log_message(f"{prefix} stderr: {stderr_data.decode('utf-8', errors='replace')}", also_print=False)

def stop_nginx(process, parent_dir):
    """Stop the nginx process"""
    if process and process.poll() is None:
        log_message("Stopping nginx")
        
        # Collect any remaining output before terminating
        try:
            stdout, stderr = process.communicate(timeout=1)
            if stdout:
                log_message(f"Nginx final stdout: {stdout.decode('utf-8', errors='replace')}", also_print=False)
            if stderr:
                log_message(f"Nginx final stderr: {stderr.decode('utf-8', errors='replace')}", also_print=False)
        except subprocess.TimeoutExpired:
            log_message("Timeout while collecting final output from Nginx")
        
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log_message("Nginx didn't terminate gracefully, killing it")
            process.kill()
        
        # Clean up pid file
        pid_file = parent_dir / "nginx.pid"
        if pid_file.exists():
            pid_file.unlink()
            log_message("Deleted nginx pid file")

def run_wrk_benchmark(url, duration=30, connections=400, threads=12):
    """Run wrk benchmark against the specified URL"""
    cmd = ["wrk", f"-t{threads}", f"-c{connections}", f"-d{duration}s", url]
    cmd_str = ' '.join(cmd)
    log_message(f"Running benchmark: {cmd_str}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        log_message(f"Error running wrk: {result.stderr}")
        return None
    
    # Log the full wrk output
    log_message("\n--- WRK Benchmark Output ---", also_print=False)
    log_message(result.stdout, also_print=False)
    log_message("--- End WRK Benchmark Output ---\n", also_print=False)
    
    return result.stdout

def parse_wrk_output(output):
    """Parse the output from wrk to extract key metrics"""
    if not output:
        return None
    
    metrics = {}
    
    # Extract requests per second
    rps_line = [line for line in output.split('\n') if "Requests/sec:" in line]
    if rps_line:
        metrics['rps'] = float(rps_line[0].split(':')[1].strip())
    
    # Extract latency
    latency_lines = [line for line in output.split('\n') if "Latency" in line]
    if latency_lines:
        parts = latency_lines[0].split()
        metrics['latency_avg'] = parts[1]
        metrics['latency_stdev'] = parts[2]
        metrics['latency_max'] = parts[3]
    
    return metrics

def start_controller(controller_path, prefix, name):
    """Start a controller process with the specified prefix"""
    if not controller_path.exists():
        log_message(f"Error: {name} controller not found at {controller_path}")
        return None
    
    cmd = [str(controller_path), prefix]
    cmd_str = ' '.join(cmd)
    log_message(f"Starting {name} controller with command: {cmd_str}")
    
    # Use binary mode for subprocess to avoid TextIOWrapper issues
    process = subprocess.Popen(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    
    # Give controller time to start
    time.sleep(2)
    
    # Check if controller is running
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        error_msg = f"Error starting {name} controller: {stderr.decode('utf-8', errors='replace')}"
        log_message(error_msg)
        log_message(f"Controller stdout: {stdout.decode('utf-8', errors='replace')}", also_print=False)
        log_message(f"Controller stderr: {stderr.decode('utf-8', errors='replace')}", also_print=False)
        return None
    
    return process

def stop_controller(process, name):
    """Stop a controller process and capture its output"""
    if process and process.poll() is None:
        log_message(f"Stopping {name} controller")
        
        # Collect any remaining output before terminating
        try:
            stdout, stderr = process.communicate(timeout=1)
            if stdout:
                log_message(f"{name} controller final stdout: {stdout.decode('utf-8', errors='replace')}", also_print=False)
            if stderr:
                log_message(f"{name} controller final stderr: {stderr.decode('utf-8', errors='replace')}", also_print=False)
        except subprocess.TimeoutExpired:
            log_message(f"Timeout while collecting final output from {name} controller")
        
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log_message(f"{name} controller didn't terminate gracefully, killing it")
            process.kill() 