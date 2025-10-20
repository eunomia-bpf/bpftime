# directly_run

A simple example to run eBPF program directly on GPU

```
make -j8
bpftime load ./directly_run &
bpftimetool run-on-cuda cuda__run
```

VecAdd (direct-run) quick start:

```
make -j8
build/tools/cli/bpftime load ./directly_run &
# optional: specify grid/block for N=1024 (4 blocks x 256 threads)
build/tools/bpftimetool/bpftimetool run-on-cuda cuda__vec_add 1 4 1 1 256 1 1
```

GEMM (direct-run) quick start:

```
make -j8
build/tools/cli/bpftime load ./directly_run &
# example: 64x64, grid=4x4, block=16x16
build/tools/bpftimetool/bpftimetool run-on-cuda cuda__gemm 1 4 4 1 16 16 1
```

Baseline vs eBPF (manual):

```
# Baseline (raw CUDA vec_add), e.g. minimal preset
benchmark/gpu/workload/vec_add 10000 10000 256 40

# eBPF direct-run vec_add (no workload binary needed)
build/tools/cli/bpftime load ./directly_run &
build/tools/bpftimetool/bpftimetool run-on-cuda cuda__vec_add 1 4 1 1 256 1 1

# Baseline (raw CUDA GEMM), e.g. minimal preset
benchmark/gpu/workload/matrixMul 32 3 32 1

# eBPF direct-run GEMM (no workload binary needed)
build/tools/cli/bpftime load ./directly_run &
build/tools/bpftimetool/bpftimetool run-on-cuda cuda__gemm 1 4 4 1 16 16 1
```

Benchmark quick run (outputs md/json/log):

```
# VecAdd benchmarks (includes baseline vs direct-run)
python3 benchmark/gpu/run_cuda_bench.py benchmark/gpu/micro/examples_vec_add_config.json
# Results: benchmark/gpu/micro/examples_vec_add_result.md/json/log

# GEMM benchmarks (baseline plus micro tests)
python3 benchmark/gpu/run_cuda_bench.py benchmark/gpu/micro/micro_gemm_config.json
# Results: benchmark/gpu/micro/micro_gemm_result.md/json/log
```
