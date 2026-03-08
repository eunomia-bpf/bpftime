# Host-side GPU map microbenchmarks

This directory contains **host-side** microbenchmarks for bpftime CUDA-backed maps.

## Prerequisites

- A working CUDA driver + toolkit (CUDA headers + `libcuda.so`)
- Build bpftime with CUDA attach enabled
- Optional (for GDRCopy speedup): install GDRCopy user library (`libgdrapi.so`) and load `gdrdrv` (creates `/dev/gdrdrv`)

## Build

From the repo root:

```bash
cmake -S . -B build -G Ninja \
  -DBPFTIME_ENABLE_CUDA_ATTACH=ON \
  -DBPFTIME_CUDA_ROOT=/usr/local/cuda \
  -DBPFTIME_ENABLE_GDRCOPY=ON

cmake --build build -j --target gpu_array_map_host_perf gpu_per_thread_array_map_host_perf
```

If you don’t want GDRCopy support, omit `-DBPFTIME_ENABLE_GDRCOPY=ON` (the binaries will still work, but `--gdrcopy 1` will always fall back to `cuMemcpyDtoH`).

## Binaries

- `build/benchmark/gpu/host/gpu_array_map_host_perf`
- `build/benchmark/gpu/host/gpu_per_thread_array_map_host_perf`

## gpu_array_map_host_perf

Benchmarks `BPF_MAP_TYPE_GPU_ARRAY_MAP` (per-key bytes = `value_size`).

```bash
# baseline (cuMemcpyDtoH)
./build/benchmark/gpu/host/gpu_array_map_host_perf \
  --iters 50000 --max-entries 1024 --value-size 8 \
  --gdrcopy 0

# enable GDRCopy (hybrid policy)
./build/benchmark/gpu/host/gpu_array_map_host_perf \
  --iters 50000 --max-entries 1024 --value-size 8 \
  --gdrcopy 1 --gdrcopy-max-per-key-bytes 4096
```

## gpu_per_thread_array_map_host_perf

Benchmarks `BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP` (per-key bytes = `value_size * thread_count`).

```bash
# baseline (cuMemcpyDtoH)
./build/benchmark/gpu/host/gpu_per_thread_array_map_host_perf \
  --iters 50000 --max-entries 1024 --value-size 8 --thread-count 32 \
  --gdrcopy 0

# enable GDRCopy (hybrid policy)
./build/benchmark/gpu/host/gpu_per_thread_array_map_host_perf \
  --iters 50000 --max-entries 1024 --value-size 8 --thread-count 32 \
  --gdrcopy 1 --gdrcopy-max-per-key-bytes 4096
```

## Flags

- `--gdrcopy 0|1`: enable/disable GDRCopy attempts
- `--gdrcopy-max-per-key-bytes <N>`: skip GDRCopy when per-key bytes `> N` (use `0` to disable the threshold)

Notes:

- If GDRCopy isn’t available at runtime (missing `libgdrapi.so` or `/dev/gdrdrv`), bpftime automatically falls back to `cuMemcpyDtoH` and performance will match baseline.

## Measuring shared_array_map overhead for issue #472

Issue #472 asks for the overhead of the server-side GPU `shared_array_map` update path. In the current codebase, the closest directly measurable host-side path is:

- `gpu_per_thread_array_map_host_perf` for `BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP`

To make the overhead meaningful, compare it against `gpu_array_map_host_perf` while keeping the effective bytes per key constant:

- plain GPU array map: `value_size = per_thread_value_size * thread_count`
- PERGPUTD array map: `value_size = per_thread_value_size`

The repository now includes a reproducible measurement script:

```bash
python3 benchmark/gpu/host/measure_shared_array_map_overhead.py \
  --build-dir build \
  --output benchmark/gpu/host/shared_array_map_overhead_rtx5090.md
```

That script:

- runs a fixed matrix of thread counts (`32`, `128`, `256`, `1024`)
- compares update and lookup cost at equal per-key bytes
- assigns a unique `BPFTIME_GLOBAL_SHM_NAME` per run so repeated measurements do not collide
- emits a markdown report ready to check into the repo

An example report generated on an RTX 5090 is checked in at:

- `benchmark/gpu/host/shared_array_map_overhead_rtx5090.md`
