# Direct runtime-library lifecycle

This Linux C++ example addresses [issue #421](https://github.com/eunomia-bpf/bpftime/issues/421).
It creates a userspace array map, an embedded eBPF counter program, a custom
event handler and a link using bpftime APIs. It loads and attaches the program,
triggers N cooperative events, reads count N, detaches, proves another trigger
does not execute, then releases the context, handlers, OS descriptors and SHM.

The application explicitly calls its event source. This is a first step toward
embedding bpftime: it does not monitor arbitrary functions in another process.
The [earlier libhelper example](https://github.com/Yinwhe/bpftime/tree/yinwhe/dev/example/bpftime-libhelper)
uses a libbpf skeleton, syscall interception and Frida agent injection for that
external-process use case. This example uses no libbpf calls, skeleton, syscall
hijacking, LD_PRELOAD, bpftime CLI subprocess, Frida or injected agent.

## Build and run

From the repository root, on Linux:

```sh
# Ubuntu build prerequisites (or equivalent packages on other distributions):
sudo apt-get install build-essential cmake ninja-build git libboost-dev zlib1g-dev python3
git submodule update --init --depth 1 third_party/Catch2 third_party/spdlog \
  third_party/argparse third_party/ubpf
git -C third_party/ubpf submodule update --init --depth 1 external/bpf_conformance
cmake -S . -B build-library -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBPFTIME_LLVM_JIT=OFF -DBPFTIME_UBPF_JIT=ON \
  -DBPFTIME_ENABLE_FRIDA=OFF -DBPFTIME_BUILD_WITH_LIBBPF=OFF \
  -DBUILD_BPFTIME_DAEMON=OFF -DBPFTIME_ENABLE_CUDA_ATTACH=OFF \
  -DBPFTIME_BUILD_RUNTIME_LIBRARY_EXAMPLE=ON -DBPFTIME_ENABLE_UNIT_TESTING=ON
cmake --build build-library --target runtime_library -j2
./build-library/example/runtime_library/runtime_library
ctest --test-dir build-library -R '^runtime_library_' --output-on-failure
```

Arguments are `[events [cycles [interpreter|jit]]]`; defaults are `5 1 interpreter`.
For example, `runtime_library 37 32 jit` repeats the full lifecycle 32 times.
Normal execution needs no root privileges or kernel eBPF permissions. The
embedded instructions avoid a BPF compiler, ELF loader and generated skeleton.

## Libraries and dependencies

The CMake target links `runtime` and `bpftime_simple_attach_impl`. Transitive
dependencies are `bpftime_vm`, `bpftime_ubpf_vm`, `libubpf.a`, `spdlog`, zlib,
and the platform thread/math/dynamic-loader libraries. Boost.Interprocess
provides the shared-memory implementation through headers.

The target also includes `bpftime_ubpf_vm_obj` directly: this retains the VM's
constructor-based factory registration, which an ordinary static-library link
can strip. A manual archive link needs equivalent registration retention.
The aggregate `libbpftime.a` alone is insufficient for this example: the simple
attach implementation is a separate component, registration must be retained,
and system link dependencies remain. Neither `bpftime-agent.so` nor
`bpftime-syscall-server.so` is required.

Catch2, argparse and uBPF's nested bpf_conformance sources are currently
repository **configuration** dependencies; this executable does not link them
and does not build the conformance runner. Python is used only by the test driver.
The reduced build disables LLVM, Frida, libbpf, the verifier, daemon and CUDA.
Other configurations may add their own link/build dependencies.

## Lifecycle and ownership

1. Generate a unique `BPFTIME_GLOBAL_SHM_NAME`, overriding any inherited name,
   and initialize with `SHM_CREATE_ONLY`. A collision fails instead of opening
   or removing someone else's session. Keep the name unchanged until cleanup.
2. Create the one-entry array and initialize its uint64 counter to zero.
3. Create a program handler from annotated instructions. The two-slot LDDW
   with `src_reg=1` resolves the map descriptor when the VM loads the program;
   helper 1 looks up key zero. The program increments the value and returns 0,
   or returns -1 if lookup fails. It does not read the custom event context.
4. Register `simple_attach_impl`, create the custom event and its link, and call
   `init_attach_ctx_from_handlers` with the SHM map helper group. This creates
   the VM, registers helpers, loads (and optionally JIT compiles), and attaches.
   Check `is_attached()` as well: a zero initialization return can mask a
   failure to instantiate an individual handler.
5. The callback passes the array's one value as the VM context (eight bytes).
   The interpreter checks loads/stores against context/stack memory; this makes
   the map helper's returned pointer accessible without disabling bounds
   checking. This example uses no separate event payload. Call the cooperative
   trigger and copy the map result while it is alive.
6. Call `destroy_instantiated_attach_link(link_fd)` while programs and maps
   still exist. Verify `is_attached()==false`, a trigger returns 1, and the
   counter is unchanged. `reset_instantiated_state()` alone does not detach.
7. Release VM/attach state, then remove handlers with `bpftime_close()` and
   close their allocated OS descriptors with `::close()`. Handler removal can
   cascade from a link to its event; it does not close the underlying OS fds.
8. `bpftime_destroy_global_shm()` unmaps this process's runtime;
   `bpftime_remove_global_shm()` unlinks its uniquely named segment.
   The session guard also cleans up after an exception, including registry
   allocation failure after exclusive SHM creation. An existing-name error
   never grants ownership and never unlinks that segment.

The runtime uses one global SHM session per process. Concurrent **process**
instances are isolated by their generated names; concurrent sessions in one
process are not demonstrated. Each simple attach implementation supports one
active link. Triggers and teardown here are serialized, and the plain counter
increment is not intended for concurrent callbacks. Treat the embedded program
as trusted input; the reduced build disables the optional verifier.

The tests exercise interpreter and uBPF JIT execution, zero events, repeated
global initialization/cleanup, concurrent processes with independent counts,
fd/SHM leak checks, cleanup after failed registry allocation, and preservation
of an inherited SHM name. The driver
launches the example executable; the example itself launches no subprocess.
`ctest -R '^runtime_library_'` is deliberate: legacy runtime unit tests are
excluded when libbpf is disabled.
