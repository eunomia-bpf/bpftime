# Table 1 private runtime preparation

Source: base `d6316fa`, branch `revision/table1-575`. No main bpftime or R5
source was modified. This worktree is intentionally uncommitted for root review.

`runtime-575.patch` contains only the three audited pr503 late-bootstrap /
early-registration compatibility files and the production
`attr.gpu_thread_count = thread_count` assignment from `ea9907d` (with a short
comment). LLVM JIT is pinned separately to ordinary source commit `f66cafa`,
which initializes LLVM once across threads. No other dirty runtime file was
copied. The five example changes are preserved in
`../../gpu_ext/workloads/llama.cpp/observability_overhead/revision-rq4/gpubpf-observability.patch`;
that patch includes the full-width sentinel readback and error propagation.
The runner still applies its clock-error change only to each private launchlate
build copy. The GPU BPF programs/workload are otherwise unchanged.

`preparation.json` records exact commands/results, file paths/sizes, local
submodule checkout revisions, source equality checks, patch checks and the
23 passing CPU tests. Each patch was applied using apply_patch; its tool
result is retained, followed by explicit post-application checks. Patch
artifacts omit content-index lines. No content identifier is used as evidence.

Root owns build, GPU execution and Git. The preparer performed no build/GPU
run. All seven paths still need a fresh actual preflight. Existing performance
results cannot be relabeled as this runtime. The new threadhist observer
records configured entries, actually overwritten entries, observed bytes and
full-width completion; the runner checks these against its requested count.
A legitimate zero-filled tail must pass; an unmodified sentinel tail fails.

## Root build entry (not run by preparer)

From `/home/yunwei37/workspace/gpu/gpu_ext`:

```bash
cmake -S ../bpftime-table1-575 -B ../bpftime-table1-575/build-table1-575 -G Ninja \\
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_EBPF_VERIFIER=OFF \\
  -DBPFTIME_ENABLE_CUDA_ATTACH=ON -DBPFTIME_LLVM_JIT=ON \\
  -DBPFTIME_ENABLE_UNIT_TESTING=OFF \\
  -DBPFTIME_CUDA_ROOT:PATH=/usr/local/cuda-12.9 \\
  -DLLVM_DIR=/usr/lib/llvm-15/cmake
cmake --build ../bpftime-table1-575/build-table1-575 --parallel 2 \\
  --target bpftime-agent bpftime-syscall-server
```

Root controls actual CPU allocation/parallelism and dependency preparation.
Debug preserves the previous pr503 build type; verification remains OFF and
this is not strict-admission evidence. CMake also builds the three default
PTX passes and the PTX compiler; keep their compiled absolute paths intact,
not just the two preload libraries. Agent and syscall server must come from
the same selected build. The statically libbpf-linked tools communicate via
syscall interception; kernelretsnoop additionally requires the exported
`bpftime_syscall_server__poll_gpu_ringbuf_map` function.

Use both runner options together:
`--bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-575` and
`--bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575`.
Changing only the build directory neither patches old tools nor records the
new source commit. Do not resume old raw directories with this source/runtime.
Keep the coordinator's CPU 16 available; it pins workers itself to CPUs 8–15.
