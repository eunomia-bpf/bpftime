# QNX 8.0 aarch64 Port — Phase 1 Skeleton

> Status: **scaffold only**. Cross-compiles the userspace VM + Frida uprobe agent path.
> Does **not** yet claim a verified run on a QNX target (Frida QNX build + Boost sysroot still required).

## Goal

On **QNX Neutrino 8.0 / aarch64**, support:

```
Linux: clang -target bpf → JSON export
QNX:   bpftimetool import → bpftime attach <pid> → Frida uprobe → eBPF (uBPF JIT)
```

Out of scope for phase-1: syscall-server, daemon, libbpf, syscall trace, text transformer, CUDA, LLVM JIT.

## CMake entry

```bash
source /path/to/qnxsdp-env.sh
export BPFTIME_FRIDA_QNX_ROOT=/path/to/frida-qnx-arm64-out

./scripts/qnx/build-phase1.sh
# or manually:
cmake -B build-qnx \
  -DCMAKE_TOOLCHAIN_FILE=cmake/qnx8-aarch64-toolchain.cmake \
  -DBPFTIME_TARGET_QNX=ON \
  -DBPFTIME_FRIDA_QNX_ROOT=$BPFTIME_FRIDA_QNX_ROOT \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-qnx --target bpftime-agent bpftime-cli-cpp bpftimetool
```

## New / key files

| Path | Role |
|------|------|
| `cmake/qnx8-aarch64-toolchain.cmake` | QNX SDP 8.0 aarch64le toolchain |
| `cmake/PlatformQNX.cmake` | Force phase-1 option defaults |
| `cmake/frida-qnx.cmake` | Local Frida gum/core (no GitHub prebuilt) |
| `scripts/qnx/build-phase1.sh` | One-shot configure + build |
| `docs/qnx-port-phase1.md` | This document |

## Forced defaults when `BPFTIME_TARGET_QNX=ON`

- `BPFTIME_BUILD_WITH_LIBBPF=OFF`
- `BUILD_BPFTIME_DAEMON=OFF`
- `BPFTIME_LLVM_JIT=OFF` / `BPFTIME_UBPF_JIT=ON`
- Skip: syscall_trace, text_segment_transformer, nv_attach, syscall-server, benchmark, AOT

## Runtime / CLI behavior on QNX

- Agent: Frida inject only (`bpftime_agent_main`); no `__libc_start_main` / LD_PRELOAD path
- CLI: `attach` / `detach` enabled; `load` / `start` return clear errors
- Platform stubs: `platform_utils`, epoll via `bpftime_epoll.h`, `/proc/<pid>/exefile` for executable path

## Remaining work (not in this PR)

1. Produce Frida `qnx-arm64` gum + core archives and document exact layout
2. Provide Boost.Interprocess for QNX target (SDP package or qnx-ports)
3. Verify `pthread_spinlock` / Boost shm under `/dev/shmem`
4. End-to-end uprobe demo on board
5. Optional: lightweight ELF bytecode loader on-target (replace JSON import)
6. Optional: re-enable LLVM JIT with QNX-hosted LLVM

## Done definition

- [ ] `build-phase1.sh` configures without FATAL (given Frida + Boost)
- [ ] `libbpftime-agent.so`, `bpftime`, `bpftimetool` link for aarch64le
- [ ] `bpftime attach <pid>` injects agent via Frida-core
- [ ] Uprobe fires; map/ringbuf updates visible via `bpftimetool`
