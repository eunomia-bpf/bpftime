# GPU Array vs Per-Thread GPU Array Host-Side Overhead

This file compares two host-side map benchmarks:

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
This comparison does not directly isolate the runtime path named `shared_array_map`.

| thread_count | effective bytes/key | gpu_array update ns/op | per_thread update ns/op | update ratio | gpu_array lookup ns/op | per_thread lookup ns/op | lookup ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 256 | 3835.9 | 3832.8 | 0.999x | 4865.1 | 4835.0 | 0.994x |
| 128 | 1024 | 3855.5 | 3851.5 | 0.999x | 4905.0 | 4866.6 | 0.992x |
| 256 | 2048 | 3903.6 | 3911.0 | 1.002x | 5044.1 | 5084.2 | 1.008x |
| 1024 | 8192 | 4325.8 | 4327.6 | 1.000x | 5843.0 | 5896.6 | 1.009x |

## Interpretation

- On `NVIDIA GeForce RTX 5090`, the PERGPUTD host-side update path is effectively on par with the plain GPU array map when normalized to the same per-key bytes.
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
python3 benchmark/gpu/host/measure_gpu_array_vs_per_thread_overhead.py \
  --build-dir build \
  --output benchmark/gpu/host/gpu_array_vs_per_thread_overhead_rtx5090.md
```

