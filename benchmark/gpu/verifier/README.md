# GPU Verifier Performance Benchmark

This benchmark measures GPU SIMT verifier latency on synthetic eBPF programs.

## What It Measures

- `program_size`: total eBPF instructions in the synthetic program
- `verify_time_us`: median verifier time in microseconds across 100 runs

Each synthetic program is a straight-line `MOV` + `ADD` sequence with a final `EXIT`. The benchmark calls `bpftime_gpu_verify_perf`, which times `bpftime::verifier::gpu::verify_gpu_program()` internally so process startup overhead does not distort the measurement.

## Build

Configure the repository with verifier support:

```sh
cmake -S . -B build -G Ninja -DENABLE_EBPF_VERIFIER=YES -DCMAKE_BUILD_TYPE=Release
cmake --build build --target bpftime_gpu_verify_perf
```

If `build/` is already configured, the Python script can build `bpftime_gpu_verify_perf` automatically.

## Run

From the repository root:

```sh
python3 benchmark/gpu/verifier/bench_verify_perf.py
```

Optional flags:

```sh
python3 benchmark/gpu/verifier/bench_verify_perf.py \
  --runs 100 \
  --sizes 16 32 64 128 256 512 1024 2048 4096 \
  --out-dir benchmark/gpu/verifier
```

## Outputs

The script writes these files in `benchmark/gpu/verifier/` by default:

- `verify_perf_results.md`
- `verify_perf_results.json`
- `verify_perf_results.csv`

The JSON and CSV rows share the same schema:

```text
program_size, verify_time_us
```

## Expected Results

Typical GPU programs should verify in under 1 ms in a `Release` build. Straight-line programs in the tens to low hundreds of instructions should stay well below that threshold; the 2048- and 4096-instruction cases are stress cases for scaling rather than representative production probes.
