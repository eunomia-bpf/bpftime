import subprocess
import re
import statistics
import time

def extract_times(output):
    """Extract the benchmark times from command output."""
    times = []
    for line in output.decode().split('\n'):
        if 'Average time usage' in line:
            # Extract the number before 'ns'
            time = float(re.search(r'Average time usage (\d+\.\d+)', line).group(1))
            times.append(time)
    return times

def run_benchmark(cmd, iterations=100):
    """Run the benchmark command multiple times and collect results."""
    all_times = []
    for _ in range(iterations):
        result = subprocess.run(cmd, shell=True, capture_output=True)
        times = extract_times(result.stdout)
        # time.sleep(0.5)
        all_times.append(times)
    return all_times

def main():
    # Commands to run
    mpk_cmd = "LD_PRELOAD=build-mpk/runtime/agent/libbpftime-agent.so benchmark/test"
    normal_cmd = "LD_PRELOAD=build/runtime/agent/libbpftime-agent.so benchmark/test"
    
    print("Running benchmarks (100 iterations each)...")
    
    # Run benchmarks
    mpk_results = run_benchmark(mpk_cmd)
    normal_results = run_benchmark(normal_cmd)
    
    # Test names
    tests = ["uprobe_uretprobe", "uretprobe", "uprobe"]
    
    print("\nResults Summary:")
    print("-" * 60)
    print(f"{'Test Name':<20} {'MPK (ns)':<15} {'Normal (ns)':<15} {'Difference':<15}")
    print("-" * 60)
    
    # Calculate and display statistics for each test
    for i, test in enumerate(tests):
        mpk_times = [result[i] for result in mpk_results]
        normal_times = [result[i] for result in normal_results]
        
        mpk_avg = statistics.mean(mpk_times)
        normal_avg = statistics.mean(normal_times)
        diff = mpk_avg - normal_avg
        diff_percent = (diff / normal_avg) * 100
        
        print(f"{test:<20} {mpk_avg:>8.2f}      {normal_avg:>8.2f}      {diff:>+8.2f} ({diff_percent:>+.1f}%)")
        
        # Print detailed statistics
        print(f"  MPK stddev: {statistics.stdev(mpk_times):.2f}")
        print(f"  Normal stddev: {statistics.stdev(normal_times):.2f}")
        print()

if __name__ == "__main__":
    main() 