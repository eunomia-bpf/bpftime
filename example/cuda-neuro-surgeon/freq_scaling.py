#!/usr/bin/env python3
import csv
import subprocess
import sys
import time
# ---- CONFIGURATION ----
GPU_ID     = "0"         # ID of the GPU to tune
DATA_FILE  = "data.csv"  # CSV file: timestamp, val1, val2, metric
METRIC_COL = 2           # zero-based index of the metric column
# -----------------------

def query_supported_clocks(gpu_id):
    """
    Return two sorted lists:
      mem_clocks  — supported memory frequencies (MHz)
      core_clocks — supported graphics (SM) frequencies (MHz)
    """
    out = subprocess.check_output([
        "nvidia-smi", "-i", gpu_id,
        "--query-supported-clocks=memory,graphics",
        "--format=csv,noheader,nounits"
    ]).decode().strip().splitlines()
    pairs = [tuple(map(int, line.split(','))) for line in out]
    mems = sorted({mem for mem, gr in pairs})
    cores = sorted({gr  for mem, gr in pairs})
    return mems, cores

def map_to_clock(val, v_min, v_max, c_min, c_max):
    """Linearly map val ∈ [v_min, v_max] to [c_min, c_max]."""
    if v_max == v_min:
        return c_min
    ratio = (val - v_min) / (v_max - v_min)
    return int(c_min + ratio * (c_max - c_min))

def load_metrics(data_file, metric_col):
    """Load CSV and return list of (timestamp, metric)."""
    records = []
    with open(data_file) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) <= metric_col:
                continue
            try:
                metric = float(row[metric_col])
                records.append((row[0], metric))
            except ValueError:
                continue
    if not records:
        print(f"No valid metrics in {data_file}", file=sys.stderr)
        sys.exit(1)
    return records

def main():
    records = load_metrics(DATA_FILE, METRIC_COL)
    metrics = [m for _, m in records]
    min_m, max_m = min(metrics), max(metrics)

    mem_clocks, core_clocks = query_supported_clocks(GPU_ID)
    m_min, m_max = mem_clocks[0],  mem_clocks[-1]
    c_min, c_max = core_clocks[0], core_clocks[-1]

    for timestamp, metric in records:
        raw_mem  = map_to_clock(metric, min_m, max_m, m_min, m_max)
        raw_core = map_to_clock(metric, min_m, max_m, c_min, c_max)
        tgt_mem  = min(mem_clocks,  key=lambda x: abs(x - raw_mem))
        tgt_core = min(core_clocks, key=lambda x: abs(x - raw_core))

        try:
            subprocess.run([
                "nvidia-smi", "-i", GPU_ID,
                "-ac", f"{tgt_mem},{tgt_core}"
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"{timestamp}: set mem={tgt_mem}MHz, core={tgt_core}MHz")
            time.sleep(20)
        except subprocess.CalledProcessError as e:
            print(f"Error at {timestamp}: {e.stderr.decode().strip()}", file=sys.stderr)

if __name__ == "__main__":
    main()