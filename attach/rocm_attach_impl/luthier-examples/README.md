# Luthier runtime verification

Build the verifier from bpftime's CMake tree, then run it on an AMD GPU:

```bash
git submodule update --init attach/rocm_attach_impl/luthier
cmake -S . -B build \
  -DBPFTIME_BUILD_LUTHIER_EXAMPLES=ON \
  -DBPFTIME_ROCM_ROOT=/opt/rocm/core-10.1 \
  -DBPFTIME_LUTHIER_ARCH=gfx1201
cmake --build build --target bpftime_luthier_probe -j8
./build/attach/rocm_attach_impl/luthier-examples/bpftime_luthier_probe
```

Building the target compiles its HIP workload and instrumentation library
dependencies. GPU execution and verification happen only when the executable
runs. No arguments are required; failure returns a nonzero exit code.

The program sources have separate roles:

- `probe_tool.cpp`: the verifier executable and runtime result checks.
- `probe_instrumentation.cpp`: the ROCprofiler tool DSO that performs Luthier
  instrumentation.
- `vector_add.hip`: an independent HIP workload with CPU result comparison;
  it contains no instrumentation IR, hooks, or Luthier dependencies.

The verifier launches the workload in a child process with an isolated profiler
environment, waits for tool finalization, and checks its output and reports.
It uses bpftime's existing header-only JSON parser. The default run covers:

| Mode | Check |
| --- | --- |
| `baseline` | Original kernel produces the expected results. |
| `unconfigured`, `bad-report` | Missing tool configuration or an unusable report path leaves the workload running without extra stdout/stderr output. |
| `roundtrip` | Luthier reconstructs and executes the binary without probes. |
| `entry` | Injected LLVM IR adds 7 to a device counter once per wave at entry. |
| `entry-exit` | Probes execute at entry and before each `S_ENDPGM`. |

Each mode checks 12 launches with two input sizes, two workgroup sizes, partial
final workgroups, and repeated dispatches. Device counters, every output
element, and tool initialization/finalization are validated. Missing GPU access
or a tool that fails to initialize is a failure.

Run one mode with `--mode entry-exit`. Optional `--application`, `--tool`, and
`--output-dir` arguments override the paths supplied by CMake. Reports and logs
are written below `build/attach/rocm_attach_impl/luthier-examples/results/<mode>/`.

## Dependencies

Requirements: CMake 3.23+, Ninja, ROCm HIP/HSA and ROCprofiler SDK development
packages, and LLVM/Clang development libraries. The submodule pins the
[eunomia-bpf/Luthier fork](https://github.com/eunomia-bpf/Luthier/tree/bpftime-rocm10-llvm24),
which adapts Luthier to LLVM 24 APIs, the descriptor-based HIP queue creation
path, and a static tooling build with private LLVM symbols.

By default, the build reads the installed LLVM's repository and exact commit
from `VCSRevision.h`, downloads that source archive into
`build/_deps/luthier-llvm-<owner>-<commit>/`, and extracts only:

- `llvm/include`
- `llvm/lib/Target/AMDGPU`
- `llvm/utils/TableGen`

Subsequent builds reuse the cache. LLVM is neither modified nor rebuilt.
For offline builds, set `BPFTIME_LUTHIER_LLVM_SOURCE` to a matching source tree
containing `llvm/`. Leave it empty for automatic download; supply it manually
if the installed LLVM does not record a GitHub repository and full commit.

The verifier is a regular `add_executable` target using bpftime's C++ compiler.
Its HIP and Luthier dependencies use an isolated CMake `ExternalProject` to
keep ROCm's compiler and LLVM separate from bpftime's host LLVM.
`BPFTIME_LUTHIER_JOBS` controls dependency build parallelism (default 8).
The build option defaults to OFF and is independent of `BPFTIME_ENABLE_ROCM_ATTACH`.

## Validation and scope

Validated outside the sandbox with package ROCm 10.1, LLVM 24, ROCprofiler SDK
1.4.1, and a Radeon RX 9070 XT (`gfx1201`, wave32): all six runtime checks pass.
Each instrumented mode observes 978 waves; entry/exit produces 6,846 in each
counter, with the same application checksum as baseline (`2349641562`).

These are synchronous experiments. Exit means wave termination, not each lane's
C++ `return`. Luthier inserts/inlines the LLVM IR payload; the example does not
splice a separately linked HSACO function into the kernel. The IR stands in for
a future eBPF compiler.

The tool uses an explicitly configured report file and forwards recoverable
failures before submission. Upstream Luthier still has fatal-error paths.
Arbitrary kernels, transparent concurrency, eBPF helpers/maps, and production
agent failure guarantees remain outside this example's scope.
