#!/usr/bin/env python3
import csv
import subprocess
from statistics import mean

# ----  CONFIGURE  ----
GPU_ID    = "0"             # which GPU to tune
DATA_FILE = "data.csv"      # your CSV of: timestamp, val1, val2, metric
METRIC_COL = 3              # zero-based index of the column you want to use as load
# ----------------------

def query_supported_clocks(gpu_id):
    """Return sorted lists of supported (graphics, memory) clocks."""
    out = subprocess.check_output([
        "nvidia-smi",
        "-i", gpu_id,
        "--query-supported-clocks=graphics.clock,mem.clock",
        "--format=csv,noheader,nounits"
    ]).decode().strip().splitlines()

    pairs = [tuple(map(int, line.split(","))) for line in out]
    cores = sorted({c for c, m in pairs})
    mems  = sorted({m for c, m in pairs})
    return cores, mems

def map_to_clock(val, v_min, v_max, c_min, c_max):
    """Linear map val∈[v_min,v_max] to [c_min,c_max]."""
    if v_max == v_min:
        return c_min
    ratio = (val - v_min) / (v_max - v_min)
    return int(c_min + ratio * (c_max - c_min))

# 1) load your metrics
records = []
with open(DATA_FILE) as f:
    reader = csv.reader(f)
    for row in reader:
        # e.g. row = ["2015-12-31 05:00","0.0089","0.0099","-0.0602"]
        try:
            metric = float(row[METRIC_COL])
            records.append((row[0], metric))
        except ValueError:
            continue

metrics = [m for _, m in records]
min_m, max_m = min(metrics), max(metrics)

# 2) find supported clocks
core_clocks, mem_clocks = query_supported_clocks(GPU_ID)
c_min, c_max = core_clocks[0], core_clocks[-1]
m_min, m_max = mem_clocks[0], mem_clocks[-1]

# 3) iterate your time points and tune
for timestamp, metric in records:
    # map metric to target clocks
    tgt_core = map_to_clock(metric, min_m, max_m, c_min, c_max)
    tgt_mem  = map_to_clock(metric, min_m, max_m, m_min, m_max)
    # snap to nearest supported
    tgt_core = min(core_clocks, key=lambda x: abs(x - tgt_core))
    tgt_mem  = min(mem_clocks,  key=lambda x: abs(x - tgt_mem))
    # apply
    cmd = [
        "nvidia-smi",
        "-i", GPU_ID,
        "-ac", f"{tgt_mem},{tgt_core}"
    ]
    subprocess.run(cmd, check=True)
    print(f"{timestamp}: set mem={tgt_mem}MHz, core={tgt_core}MHz")
    # if you want to wait until the next timestamp, 
    # you could compute a sleep here based on real clock.
