# CUTLASS GPU Demo

This directory hosts a complete GPU attach demo that marries a heavy CUTLASS GEMM workload with a matching eBPF launch counter. By dialing the matrix shape and launch count you can go from quick smoke tests to kernels that resemble production GEMM/BatchGEMM loads (LLM-style 4K–8K problems).

- `gemm/`: C++ driver that instantiates `cutlass::gemm::device::Gemm<float,...>` and lets you configure matrix sizes, warmups, launch count, RNG seed, and optional CPU verification.
- `counter/`: eBPF kretprobe (`cutlass_launch_counter`) that attaches to the CUTLASS kernel and increments a GPU array map every time the kernel exits.

Everything below replaces the subdirectory READMEs, so you only need this file for build/run instructions.

## Build

```bash
# from repo root
make -C example/gpu/cutlass CUTLASS_DIR=.../cutlass
```

- `CUTLASS_DIR` is optional; if omitted the Makefile clones CUTLASS v3.5.0 into `.deps/cutlass`.
- Build bpftime itself with CUDA attach enabled (`-DBPFTIME_ENABLE_CUDA_ATTACH=ON`) so the syscall-server / agent shims exist.

## Workload Options (`gemm/cutlass_gemm`)

```
cutlass_gemm [--shape MxNxK] [--m M] [--n N] [--k K]
             [--launches N] [--warmup N] [--seed S] [--verify]
```

- `--shape` is shorthand for M/N/K (default `4096x4096x4096`).
- `--launches` controls the number of timed kernel launches after warmup (default `24`).
- `--warmup` runs extra launches before timing (default `2`).
- `--verify` enables a CPU reference check for matrices ≤512³; larger shapes skip the check automatically.
- `--seed` reproducibly seeds the host RNG (default `42`).

At the end of a run the binary prints total/average time, an approximate GFLOP/s rate, plus checksum and L2 norm of the result tensor so you can tell runs apart when sweeping configurations.

## Running the Demo

Run the counter (server) and workload from two terminals:

1. **Terminal A – counter under syscall-server shim**

```bash
BPFTIME_LOG_OUTPUT=console \
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
   example/gpu/cutlass/counter/cutlass_launch_counter
```

   The counter prints timestamps with the cumulative launch count (e.g. `12:34:56 CUTLASS launches: 84`).

2. **Terminal B – CUTLASS workload under the agent shim**

```bash
BPFTIME_LOG_OUTPUT=console \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
   example/gpu/cutlass/gemm/cutlass_gemm --shape 4096x4096x4096 --launches 24 --warmup 2
```

For quick validation switch to something lighter, e.g. `--shape 1024x1024x1024 --launches 4 --verify`.

As soon as the workload starts running, terminal A’s counter reflects each kernel exit via the GPU array map, proving that bpftime’s CUDA attach path handles large, manually constructed PTX.

### Sample Output

Terminal A (server):

```
12:34:56 CUTLASS launches: 24
12:34:57 CUTLASS launches: 48
12:34:58 CUTLASS launches: 72
```

Terminal B (workload) for the default heavy case:

```
[cutlass] problem: 4096x4096x4096 | launches=24 (warmup=2) | host seed=42
[cutlass] running 2 warmup launches...
[cutlass] running timed workload...
[cutlass] completed 24 launches in 551.21 ms (avg 22.97 ms) | 5984.17 GFLOP/s
[cutlass] checksum=4631.49 | l2-norm=87387.3
```

Numbers vary slightly per GPU, but you should always see the counter climbing in step with the workload’s launch count and the GEMM binary printing the timing/GFLOP/s summary.

## Customizing the Target Kernel

`counter/cutlass_launch_counter.bpf.c` names the kernel produced by the default `cutlass_gemm` template instantiation. If you change the template parameters (tile shapes, op class, accumulation type, etc.), update the SEC annotation:

1. Rebuild the workload (`make -C example/gpu/cutlass/gemm CUTLASS_DIR=.../cutlass`).
2. Extract the mangled kernel symbol:

   ```bash
   cuobjdump --dump-elf-symbols example/gpu/cutlass/gemm/cutlass_gemm | grep Gemm
   ```

3. Replace the SEC string inside `counter/cutlass_launch_counter.bpf.c` with the new name and rebuild the counter via `make -C example/gpu/cutlass counter`.

That is all—this README now documents both halves (workload + counter) so you can delete any local copies of the old per-directory instructions.
