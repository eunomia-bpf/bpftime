# eBPF probes on ROCm kernels

This independent example runs real eBPF through llvmbpf, Luthier and
`bpftime start --rocm`. The HIP application contains no instrumentation code.

| Source | Purpose |
| --- | --- |
| `rocm_ebpf.cpp` | Loads probes, displays Array values, or checks CPU/GPU request-response rounds. |
| `rocm_ebpf.bpf.c` | Entry/return probes using Array map helpers and `bpf_test_println`. |
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

The loader displays the same shared Array map that the GPU probes access:

```text
Array: entry=2 exit=2 entry_value=123 exit_value=124 errors=0
```

Entry/exit counters use atomic increments through the pointer returned by
`bpf_map_lookup_elem`. The two values demonstrate `bpf_map_update_elem`.
`errors` counts unexpected results for updates, out-of-range keys,
`BPF_NOEXIST` and deletion (Array deletion returns `-EINVAL`). Counts accumulate
while the loader remains running; another `victim 8 3 2` adds 48 to each
enabled counter. Intermediate snapshots may show one probe ahead of the other.

The victim also accepts `blocks repeats streams`, for example `victim 8 3 2`
launches 48 waves across two streams. Running `./example/rocm-ebpf/victim`
directly executes the same kernel without bpftime instrumentation.

## Bidirectional map test

Use the same environment settings from **Run**, and select `roundtrip` in the
first terminal:

```sh
./build/tools/cli/bpftime --install-location "$PWD/build" load \
  ./example/rocm-ebpf/rocm_ebpf roundtrip 20
```

In the second terminal, run one persistent victim with twenty spaced launches:

```sh
./build/tools/cli/bpftime --install-location "$PWD/build" start --rocm \
  ./example/rocm-ebpf/victim 1 20 1 100
```

The last victim argument is the delay in milliseconds between completed launch
batches (default zero). It gives the CPU time to publish the next request.
The victim still contains no BPF or map access code.

The `mailbox` Array has three 64-bit slots:

| Slot | Writer | Reader | Value |
| --- | --- | --- | --- |
| 0 | CPU loader, using `bpf_map_update_elem` | GPU entry/return probes | A new input for each round, starting above 2³² |
| 1 | GPU entry probe, using the update helper | CPU loader | `input * 3 + 1` |
| 2 | GPU return probe, using an atomic exchange through the lookup pointer | CPU loader | `input * 7 + 2` |

The CPU checks both replies before publishing the next input. Changing the
input every round prevents an old response from satisfying a later round and
tests CPU writes made **after** GPU instrumentation and memory mapping are
already active. Multiple waves can respond; the test counts completed requests,
not waves. An older in-flight wave may write an older response, so the loader
polls until both slots match the current request.

Example loader output:

```text
CPU -> GPU round=1 input=4294967313
GPU -> CPU round=1 entry=12884901940 exit=30064771193 OK
...
ROUNDTRIP OK rounds=20
```

The loader exits with status 0 only after every round passes. A map syscall
failure, interruption or missing/mismatched reply for 30 seconds returns status
1. `roundtrip` defaults to three rounds; start the victim within the first
30 seconds. For concurrent waves/streams, use `victim 4 20 4 50` instead.
Normal `entry`, `exit` and `both` modes retain their original probes.

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

This subset passes `nullptr` as BPF context. It supports stack-local data,
helper 506, and helpers 1/2/3 on ordinary `BPF_MAP_TYPE_ARRAY` maps. Map fds are
resolved to channel handles during IR linking. A GPU system-scope spinlock
protects each complete request; the agent's CPU worker executes the existing
Array implementation and returns the result. Lookup returns the actual map
value through a fine-grained HSA mapping, so device stores and atomics update
the map visible to the loader. The channel lock does not make ordinary value
loads/stores atomic.

The agent retains Array backing storage independently of fd lifetime, so
closing/reusing an fd does not retarget an in-flight probe. The channel and
mapping stay alive until the instrumented dispatches have completed. No HIP
initialization is needed in the loader or agent constructor.

Current limits are one GPU per agent, 64 retained map bindings per channel,
4-byte keys and values up to 4096 bytes. CPU/GPU coherent memory and 32/64-bit
system atomics are required. Hash/per-CPU/GPU-native maps, program-side
read/write-restricted maps, global-data relocations and BPF-to-BPF calls are
rejected. Helpers are inlined, and their target architecture must match
`BPFTIME_LUTHIER_ARCH`. A helper waits at most five seconds for the lock and
response; timeout permanently stops that channel, returns NULL/`-ETIMEDOUT`,
and subsequent dispatches run without instrumentation. Timed-out updates are
not retried and may already have taken effect. The transport serializes
requests and is intended as a functional CPU helper path, not a GPU-native
map performance optimization.

For the backend's manual GPU timeout regression (not registered with CTest):

```sh
cmake --build build --target bpftime_rocm_channel_tests
./build/attach/rocm_attach_impl/test/rocm_channel_tests
```

Recoverable compilation/attachment errors preserve the original dispatch.
Upstream LLVM/Luthier still contain fatal compiler paths, so these checks do
not establish support for arbitrary kernels or untrusted BPF programs.
In the tested gfx1201 Debug build, a separate entry payload containing only
an Array lookup followed by an atomic increment triggers a Luthier machine
verifier error (`V_WRITELANE_B32` uses an undefined physical register) and
aborts the victim. The supplied example and roundtrip payloads pass, but
their success does not establish fail-open behavior for other payload shapes.

