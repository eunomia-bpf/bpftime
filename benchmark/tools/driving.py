import subprocess
import re
import json
import numpy as np


# Function to run the command and extract the average write time
def run_command_and_extract_time():
    print("run_command_and_extract_time")
    try:
        result = subprocess.check_output(
            [
                "sudo",
                "LD_PRELOAD=build/runtime/agent/libbpftime-agent.so",
                "benchmark/syscall/victim",
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


# Run the command 100 times and collect the average write times
times = [run_command_and_extract_time() for _ in range(20)]
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
with open("output_metrics.json", "w") as f:
    json.dump(data, f, indent=4)

print("Data has been saved to output_metrics.json")
