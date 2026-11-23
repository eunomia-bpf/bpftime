# syscall_server.so

Use as a LD_PRELOAD to intercept bpf syscalls and mock them in userspace, or trace them and make kernel eBPF run with userspace eBPF.

## Run with userspace eBPF

The default behavior is to run with userspace eBPF. Using userspace eBPF means using userspace eBPF maps in shared memory, using userspace eBPF verifier and userspace eBPF runtime.

server

```sh
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so example/malloc/malloc
```

client

```sh
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so example/malloc/victim
```

## run with kernel

Set the environment variable `BPFTIME_RUN_WITH_KERNEL` to `true` to make the kernel eBPF run with userspace eBPF. This means using kernel eBPF maps instead of userspace eBPF maps, and using kernel eBPF verifier instead of userspace eBPF verifier.

```sh
BPFTIME_RUN_WITH_KERNEL=true
```

example start tracing:

```sh
SPDLOG_LEVEL=Debug BPFTIME_RUN_WITH_KERNEL=true LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so example/malloc/malloc
```

example run target program:

```sh
SPDLOG_LEVEL=Debug LD_PRELOAD=build/runtime/agent/libbpftime-agent.so example/malloc/victim
```

## skip verification for some programs

```sh
BPFTIME_NOT_LOAD_PATTERN=.*
```

## Rootless synthetic tracepoint IDs: single source of truth

In rootless mode the tracefs event `id` files can be unreadable, so this
server mocks `.../events/<event>/id` while the attach side independently
builds its tracepoint id table. Both ends MUST derive ids from the same
source, otherwise a profiler's `perf_event_open(config=id)` cannot resolve
back to the mocked event and aborts during startup.

- `rootless::synthetic_tracepoint_id()` in
  `runtime/include/platform_utils.hpp` is the neutral single-source encoder
  (FNV-1a over the tracefs event path). It lives in a header shared through
  `runtime/include/`, so neither component needs an `attach/` dependency.
- The mock provider in `syscall_server_utils.cpp` first calls
  `rootless_synthetic_id_for_event()`, which only encodes
  `syscalls/sys_enter_*` / `syscalls/sys_exit_*`. When that returns 0 it
  falls back to `rootless::synthetic_tracepoint_id()` instead of letting the
  `openat` fail with `ENOENT`.
- The attach table in
  `attach/syscall_trace_attach_impl/src/syscall_table.cpp` builds its keys
  with the same `rootless::synthetic_tracepoint_id()`.

Any change to the id encoding MUST update both ends in the same change and
cover a rootless end-to-end attach for a non-syscall tracepoint; the
regression tests live in `runtime/unit-test/test_tracepoint_id.cpp` (#857).
Changing only one side silently turns the profiler startup into a fatal
failure.
