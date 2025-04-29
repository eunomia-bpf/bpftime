#!/bin/bash

# Run all experiments

# run micro-benchmarks

python3 benchmark/uprobe/benchmark.py
python3 benchmark/syscall/benchmark.py
python3 benchmark/mpk/benchmark.py

# run system-benchmarks
python3 benchmark/syscount-nginx/benchmark.py
python3 benchmark/ssl-nginx/draw_figture.py
# python3 benchmark/redis-durability-tuning/benchmark.py
# python3 benchmark/deepflow/benchmark.py
# python3 example/attach_implementation/benchmark/run_benchmark.py
