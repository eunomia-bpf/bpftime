# Trap uprobe backend microbenchmark

Measures per-hit latency of the trap (breakpoint + SIGTRAP) uprobe backend
on riscv64.  Analogous to `benchmark/uprobe/` but drives the trap API
directly instead of going through LD_PRELOAD.

## Build

From the repository root (riscv64 cross-build):

```sh
cmake -Bbuild-rv \
  -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBPFTIME_ENABLE_TRAP_UPROBE=ON \
  -DBPFTIME_BUILD_WITH_LIBBPF=OFF \
  -DBPFTIME_ENABLE_FRIDA=OFF \
  -DBUILD_BPFTIME_DAEMON=OFF \
  -DBPFTIME_LLVM_JIT=OFF \
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH \
  -DBoost_INCLUDE_DIR=/usr/include

cmake --build build-rv --target trap-uprobe-bench -j$(nproc)
```

On native riscv64, replace the toolchain file with a normal build.

## Run

Under qemu-user:

```sh
qemu-riscv64 -L /usr/riscv64-linux-gnu \
  build-rv/benchmark/trap-uprobe/trap-uprobe-bench [iterations] [threads]
```

The Python driver collects multiple runs and writes `results.md`:

```sh
python3 benchmark/trap-uprobe/benchmark.py --qemu --iter 50000 --runs 5 --threads 4
```

On native riscv64 hardware, omit `--qemu`.

## What it measures

| Test | Description |
|---|---|
| Baseline | Direct function call, no probe attached |
| Uprobe | One uprobe callback (noop) on the target |
| Uretprobe | One uretprobe callback (noop) on the target |
| Uprobe + Uretprobe | Both on the same target |
| Multi-thread | N threads calling a probed function concurrently |

Each target function is `a[b] + c` — trivial, so the measured time is
dominated by probe overhead.

## Comparing with kernel uprobe

On native riscv64 with kernel eBPF support, build the kernel uprobe
benchmark from `benchmark/uprobe/` and run both side by side.  The expected
result is that the trap backend is faster (no kernel/user context switch)
for pure uprobe/uretprobe hits, similar to the ~10x speedup seen on x86_64
with the frida backend.

## Example results (qemu-user)

Note: qemu signal emulation adds ~50-100x overhead; these numbers are NOT
representative of real hardware.

| Probe type | Avg (ns) | Overhead (ns) |
|---|---:|---:|
| Baseline | 65 | — |
| Uprobe | 6185 | 6119 |
| Uretprobe | 9522 | 9457 |
| Uprobe + Uretprobe | 9993 | 9928 |
