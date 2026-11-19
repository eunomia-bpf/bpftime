# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

bpftime is a high-performance userspace eBPF runtime and general extension framework. It enables running eBPF programs in userspace for observability, networking, GPU, and other use cases, achieving up to 10x better performance than kernel eBPF for certain operations.

## Common Development Commands

### Build Commands

```bash
# Build with tests and all components (Debug mode)
make build

# Build release version
make release

# Build with LLVM JIT support
make release-with-llvm-jit

# Build with static library
make release-with-static-lib

# Clean build artifacts
make clean

# Build specific components
cmake --build build --target <target_name>
```

### Testing Commands

```bash
# Run all unit tests
make unit-test

# Run runtime tests only
make unit-test-runtime

# Run daemon tests only
make unit-test-daemon

# Build eBPF test programs (from runtime/test/bpf)
make -C runtime/test/bpf

# Run specific test binary
./build/runtime/unit-test/bpftime_runtime_tests
./build/daemon/test/bpftime_daemon_tests
```

### Running Examples

```bash
# Build example programs
make -C example/malloc

# Load and run eBPF program
export PATH=$PATH:~/.bpftime/
bpftime load ./example/malloc/malloc
bpftime start ./example/malloc/victim

# Attach to running process (requires sudo)
sudo bpftime attach <pid>
```

### Development Tools

```bash
# Install to system (after build)
sudo cmake --install build

# Build with specific features
cmake -Bbuild -DBPFTIME_ENABLE_IOURING_EXT=1  # IO uring support
cmake -Bbuild -DBPFTIME_LLVM_JIT=1           # LLVM JIT backend
cmake -Bbuild -DBPFTIME_ENABLE_MPK=1         # Memory Protection Keys
```

## Architecture & Code Structure

### Component Organization

The codebase is organized into distinct components that interact through well-defined interfaces:

1. **`vm/`** - Virtual Machine implementations
   - `vm-core/`: Core VM interface and abstractions
   - `llvm-jit/`: LLVM-based JIT/AOT compiler (high performance)
   - Supports multiple backends (LLVM JIT, ubpf)

2. **`runtime/`** - Core runtime functionality
   - `include/`: Public APIs and interfaces
   - `src/handler/`: Handler pattern for managing eBPF objects (programs, maps, links)
   - `src/bpf_map/`: Map implementations (hash, array, ringbuf, etc.)
   - `syscall-server/`: Intercepts eBPF syscalls for compatibility
   - `agent/`: Library injected into target processes

3. **`attach/`** - Event attachment mechanisms
   - `base_attach_impl/`: Abstract interface for all attach types
   - `frida_uprobe_attach_impl/`: Dynamic instrumentation via Frida
   - `syscall_trace_attach_impl/`: Syscall interception
   - `nv_attach_impl/`: CUDA/GPU event attachment

4. **`daemon/`** - System monitoring and kernel interaction
   - Monitors kernel eBPF operations
   - Can redirect to userspace runtime

5. **`bpftime-verifier/`** - Safety verification
   - Wraps PREVAIL verifier for userspace
   - Optional kernel verifier integration

### Key Architectural Patterns

1. **Shared Memory Architecture**: Uses boost::interprocess for zero-copy IPC between processes
   - Central `handler_manager` registry in shared memory
   - File descriptor abstraction for kernel compatibility

2. **Handler Pattern**: All eBPF objects (programs, maps, links) managed through handlers
   - Consistent interface for create/destroy/access
   - Enables cross-process object sharing

3. **Modular Attach System**: New event sources added by implementing `base_attach_impl`
   - Each attach type provides its own helpers
   - Supports custom context preparation

4. **VM Abstraction**: Multiple VM backends through common interface
   - Switch between interpreter/JIT/AOT transparently
   - Extensible helper function registration

### Important Files and Interfaces

- `runtime/include/bpftime.hpp`: Main runtime API
- `runtime/include/bpftime_shm.hpp`: Shared memory management
- `vm/vm-core/include/ebpf-vm.h`: VM interface
- `attach/base_attach_impl/base_attach_impl.hpp`: Attach interface
- `runtime/src/handler/handler_manager.hpp`: Central object registry

### Development Workflow

1. **Adding new functionality**: Check existing patterns in similar components
2. **Testing**: Add unit tests in component's test directory
3. **eBPF programs**: Use standard clang/libbpf toolchain, test with both userspace and kernel
4. **Performance**: Consider shared memory overhead, prefer batch operations

The architecture prioritizes performance (bypassing kernel), compatibility (same APIs as kernel eBPF), and extensibility (modular design for new features).

## Host-process safety invariants

Code loaded into a third-party process through `LD_PRELOAD`, runtime
injection, or a breakpoint/trap is a guest of that process. The host process
must keep running with its own semantics whenever bpftime itself fails. Every
invariant below is a review requirement: a change that weakens one must update
the tracking entry and the regression coverage in the same pull request.

1. **Never terminate the host.** No bpftime initialization, configuration,
   allocation, attach, or execution path may call `exit`, `_exit`, `abort`,
   `terminate`, or otherwise end the host because of an internal error.
   Return or record the error and disable only the affected bpftime
   operation. No C++ exception may cross an exported C or interposition
   boundary; catch at the boundary and fail open. Tracked residual sites are
   collected under issue #605.
2. **Fail open to the host operation.** When an interposed call, shim, or
   attach cannot be serviced, fall back to the original libc/syscall
   behavior (or skip the affected probe) instead of failing the host call.
3. **Leave host standard streams untouched.** Do not write to, close,
   redirect, or flush the host's stdin, stdout, or stderr unless the user
   explicitly selected console output (`BPFTIME_LOG_OUTPUT=console`). Route
   diagnostics and `bpf_trace_printk` through the configured bpftime sink;
   a sink setup failure falls back to a silent sink. A residual
   initialization-logging path is tracked under issue #692.
4. **Preserve host signal disposition.** Never replace a host signal handler
   without chaining to it or restoring the previous disposition. Trap/SIGTRAP
   handlers forward signals that do not originate from our breakpoints.
   Process-wide SIGSEGV handling for protected memory copies is opt-in,
   scoped to the protected-copy window, and rethrows faults that occur
   outside the window to the previous disposition.
5. **Fault-safe host memory access.** Helpers that dereference host pointers
   (`bpf_probe_read*`, `bpf_probe_write_user`, stack unwinding) must never
   dereference an unchecked pointer. A safe backend (`process_vm_readv`/
   `process_vm_writev`, a scoped signal-recovery backend, or a bounded stack
   snapshot) must be selected at build time; the bare-`memcpy` backend is an
   explicit, named opt-in for controlled benchmarks, never a silent default.
   Faults are reported as `-EFAULT`, never as a host crash. See the
   "Host memory access safety" section of `runtime/README.md`.
6. **Validate attach inputs against mapped extents.** Resolved hook
   addresses must be bounds-checked against the target module's mapped
   ranges before a trampoline is installed; malformed attach specs are
   rejected through the normal failure path. Tracked under issue #716.
7. **Admission of untrusted programs.** eBPF programs are externally
   supplied code. Where a verifier or helper whitelist exists, programs
   that fail it are rejected at load/link time; a rejected helper is never
   replaced with a silent no-op. JIT compilation required for
   signal-context execution finishes before a probe is armed. The verifier
   is currently opt-in (`ENABLE_EBPF_VERIFIER`, default OFF) and
   kprobe-shaped sections are not yet verified; this gap is tracked in
   issue #699, and deployments that cannot verify probes must treat them
   as trusted.
8. **Restore global host state and release resources.** CPU affinity,
   signal masks, page permissions, and dynamic linker state touched by
   bpftime are restored on every exit path. Shared-memory ownership must
   leave no `/dev/shm` residue across process exit orders; tracked under
   issue #701.
9. **Regression coverage is part of the invariant.** Each invariant needs
   a test that fails when it is violated: host exit status and default
   stdio silence under shm exhaustion, bad-pointer survival for every
   probe-read backend, signal chaining, attach bounds, and admission
   rejection. CI must observe the instrumented victim's real exit code;
   a victim segfault is a hard failure, never a retried flake
   (issue #743).

## Trap and RISC-V backend constraints

Trap callbacks execute inside SIGTRAP. Runtime-managed BPF links must reject
programs whose concrete helper implementations have not been audited as
async-signal-safe, and must finish JIT compilation before arming a probe.
Other attach backends retain their existing helper support. Never replace a
rejected helper with a silent no-op. Native/opaque callbacks are responsible
for satisfying the same signal-safety contract themselves.

RISC-V trap validation must include a real 4-byte instruction at an address
congruent to 2 modulo 4, with assertions on both instruction width and the
three-phase write counter. Test late dlopen after worker threads already exist,
and permission failures during arm, disarm and rollback. QEMU results do not
establish native concurrent patch/restore behavior.

## Repository Documentation

Keep cross-cutting constraints in this file; keep detailed component design
next to the code in that component's `README.md`
(see `attach/trap_uprobe_attach_impl/README.md` for the established format,
including its Host process contract section). Do not create a root `docs/`
directory and do not commit pull-request review artifacts. The main README
stays user-facing and does not host maintenance invariants.

## Syscall Interception and Seccomp Layering

Syscall interception has three independent layers. Keep them physically
separate; do not merge mechanisms that intercept at different points.

1. **`runtime/syscall-server/` - the fake kernel.** It owns the only
   fd/mmap resource ledger and emulated syscall state. Anything it does not
   emulate must fall through to the real syscall (`call_original()`). It does
   not include any `attach/` header and must stay buildable without the agent.
   When it needs seccomp, it implements its own small, self-contained seccomp
   hook inside this directory; it must not reach into an attach implementation.
2. **`attach/` - BPF event sources.** Each attach mechanism is a module under
   `attach/` that is linked into `bpftime-agent` (see
   `runtime/agent/CMakeLists.txt`); attach implementations are not standalone
   shared objects. A seccomp-based attach mechanism lives in its own
   `attach/seccomp_attach_impl/` module and is linked into the agent like every
   other attach impl. It produces events for BPF programs and reaches the fake
   kernel only through the `get_call_hook()` / dispatcher function-pointer seam,
   never by linking `bpftime-syscall-server` directly.
3. **`attach/text_segment_transformer/` - Frida/libc binary rewriting only.**
   It exists solely because the Frida trampoline must run before the agent and
   hijack `__libc_start_main`. Seccomp code does not belong here; do not add
   seccomp filters, workers, or resource tables to this module.

The two seccomp implementations are deliberately separate because they serve
different masters: the syscall-server version intercepts to emulate resources
without an agent; the attach version intercepts to feed BPF callbacks inside
the agent. Do not unify them into one shared implementation.

Cross-layer coupling is allowed only through the existing function-pointer
seam (`get_call_hook`/`set_call_hook`). No new link dependency or `#include`
may cross the syscall-server / attach boundary.

## Change Size and Scope Discipline

- Make the smallest change that satisfies the task. Prefer deleting dead or
  duplicated code over adding new code.
- Do not modify files unrelated to the change. A refactor that moves a module
  must touch only that module, its build target, and the docs describing it;
  leave neighboring modules, formatting, and unrelated call sites untouched.
- Do not duplicate state or helpers that already exist in another layer. In
  particular, never keep a second fd/mmap ledger outside `syscall-server`.
- Before introducing a structure, enum, or register-layout type, check for an
  existing shared definition and reuse it instead of declaring a parallel one.
