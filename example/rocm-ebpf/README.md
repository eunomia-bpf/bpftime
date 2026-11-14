# eBPF probes on ROCm kernels

This independent example runs real eBPF through llvmbpf, Luthier and
`bpftime start --rocm`. The HIP application contains no instrumentation code.

| Source | Purpose |
| --- | --- |
| `rocm_ebpf.cpp` | Userspace eBPF program: loads and attaches probes, then waits until Ctrl-C. |
| `rocm_ebpf.bpf.c` | Entry and return probes calling `bpf_test_println`. |
| `victim.hip` | HIP application containing the kernel to instrument. |

## Build

First build bpftime with ROCm attach and the CLI enabled, using the same ROCm
LLVM installation for llvmbpf and Luthier. From the repository root, this is
the configuration tested with package ROCm 10.1, LLVM 24 and a Radeon RX 9070 XT
(`gfx1201`):

```sh
cmake -S . -B build \
  -DBPFTIME_ENABLE_ROCM_ATTACH=ON \
  -DBPFTIME_ENABLE_CUDA_ATTACH=OFF \
  -DBPFTIME_ENABLE_FRIDA=ON \
  -DBPFTIME_ROCM_ROOT=/opt/rocm/core-10.1 \
  -DBPFTIME_LUTHIER_ARCH=gfx1201
cmake --build build --target bpftime-agent bpftime-syscall-server bpftime-cli-cpp -j6
```

Build this example separately with Make:

```sh
make -C example/rocm-ebpf -j6
```

The Makefile builds its own static libbpf and BPF object under `.output/`, and
places the `rocm_ebpf` and `victim` executables in the example directory.
Dependencies are a C++17 compiler, Clang with the BPF
backend, libelf/zlib development headers, and ROCm HIP. The example has no
CMake target.

`ROCM_PATH` defaults to `hipconfig --rocmpath` and `GPU_ARCH` to `native`.
Override them as needed; the GPU architecture must match the agent's
`BPFTIME_LUTHIER_ARCH`:

```sh
make -C example/rocm-ebpf -j6 CLANG=clang-20 \
  ROCM_PATH=/opt/rocm/core-10.1 GPU_ARCH=gfx1201
```

Use `make -C example/rocm-ebpf clean` before changing compiler or architecture
settings. The example's Makefile does not configure or build bpftime itself.

## Run

From the repository root, use these environment settings in **both terminals**:

```sh
export BPFTIME_GLOBAL_SHM_NAME=bpftime-rocm-example
# This example uses a custom GPU helper, outside the CPU verifier's helper set.
# Only run the supplied, trusted BPF programs with verification disabled.
export BPFTIME_VERIFIER_LEVEL=NO_VERIFY
```

In the first terminal, load the eBPF programs and leave this process running:

```sh
./build/tools/cli/bpftime --install-location "$PWD/build" load \
  ./example/rocm-ebpf/rocm_ebpf
```

In the second terminal:

```sh
./build/tools/cli/bpftime --install-location "$PWD/build" start --rocm \
  ./example/rocm-ebpf/victim
```

The victim launches two waves by default. Its output includes two `BPF entry`
lines, two `BPF exit` lines, and `VICTIM OK waves=2` after checking the numerical
results. Probe output appears in the victim process. The eBPF process accepts
`entry`, `exit` or `both` (default) to select which probes to load.

The victim also accepts `blocks repeats streams`, for example `victim 8 3 2`
launches 48 waves across two streams. Running `./example/rocm-ebpf/victim`
directly executes the same kernel without bpftime instrumentation.

## Probe contract

`rocm_ebpf.bpf.c` uses `kprobe/rocm__<kernel>` and
`kretprobe/rocm__<kernel>` sections; program names end in `__rocm` so the runtime
selects the GPU path. Kernel names match exact ELF symbols (the example uses
`extern "C"` to avoid C++ name mangling). Entry runs before the first instruction;
return runs before `S_ENDPGM`. Each probe runs in **one active lane per wave**,
not once per launch or once per work item. A lane's early C++ return is not a
separate return probe.

`bpf_test_println(const char *)` is helper **506**, declared in
[`bpf_test_println.h`](../../attach/rocm_attach_impl/bpf_test_println.h). It calls
ROCm device `printf("%s\n", message)`, backed by Luthier's hostcall service.
The pointer must reference a valid, NUL-terminated device string. Use a local
stack array as shown; `%` characters are printed literally. Output is explicitly
requested by the probe and goes to the application's stdout. This is a testing
helper, not a production logging interface.

The agent translates eBPF into target-neutral LLVM IR, links the helper bitcode,
converts stack allocations to AMDGPU private memory, and asks Luthier to inject
the payload into the loaded kernel binary. Rewritten kernels are cached by
kernel descriptor and probe generation. Completion callbacks retain extended
kernel arguments until GPU completion; detach affects subsequent dispatches.

This initial subset passes `nullptr` as BPF context. It supports stack-local
data and helper 506; maps, global-data relocations, other helpers and BPF-to-BPF
calls are rejected. The helper is inlined, and its target architecture must
match `BPFTIME_LUTHIER_ARCH`. CPU helper RPC and GPU maps are separate work.
Recoverable compilation/attachment errors preserve the original dispatch.
Upstream LLVM/Luthier still contain fatal compiler paths, so these checks do
not establish support for arbitrary kernels or untrusted BPF programs.

