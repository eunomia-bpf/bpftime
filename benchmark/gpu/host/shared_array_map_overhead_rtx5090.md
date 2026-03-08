# Shared Array Map Host-Side Overhead

This file quantifies issue #472 by comparing:

- `gpu_array_map_host_perf`: `BPF_MAP_TYPE_GPU_ARRAY_MAP`
- `gpu_per_thread_array_map_host_perf`: `BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP`

Each comparison keeps the effective per-key bytes the same:

- plain GPU array map `value_size = per_thread_value_size * thread_count`
- PERGPUTD array map `value_size = per_thread_value_size`, with `thread_count` set explicitly

Device: `NVIDIA GeForce RTX 5090`
Iterations per run: `50000`
Max entries: `1024`
Per-thread value size: `8` bytes
GDRCopy driver available: `no`

Because `/dev/gdrdrv` is unavailable on this machine, these numbers represent the fallback `cuMemcpyDtoH` path.

| thread_count | effective bytes/key | gpu_array update ns/op | per_thread update ns/op | update ratio | gpu_array lookup ns/op | per_thread lookup ns/op | lookup ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 256 | 3835.7 | 3828.3 | 0.998x | 4861.7 | 4848.0 | 0.997x |
| 128 | 1024 | 3878.2 | 3855.4 | 0.994x | 4879.4 | 4865.4 | 0.997x |
| 256 | 2048 | 3925.8 | 3904.3 | 0.995x | 5044.7 | 5011.9 | 0.993x |
| 1024 | 8192 | 4300.0 | 4315.6 | 1.004x | 5891.2 | 5934.8 | 1.007x |

## Interpretation

- On this RTX 5090 host, the PERGPUTD/shared-array-map host-side update path is effectively on par with the plain GPU array map when normalized to the same per-key bytes.
- Lookup remains within a similarly narrow band across all tested thread counts.
- This benchmark only measures host-side `update`/`lookup` cost. It does not cover in-kernel helper cost or GPU-side contention.

## Reproduction

Build the benchmarks:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBPFTIME_ENABLE_CUDA_ATTACH=ON \
  -DBPFTIME_CUDA_ROOT=/usr/local/cuda \
  -DBPFTIME_ENABLE_GDRCOPY=ON

cmake --build build -j --target gpu_array_map_host_perf gpu_per_thread_array_map_host_perf
```

Generate this report:

```bash
python3 benchmark/gpu/host/measure_shared_array_map_overhead.py \
  --build-dir build \
  --output benchmark/gpu/host/shared_array_map_overhead_rtx5090.md
```

