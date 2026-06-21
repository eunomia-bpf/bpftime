# OTel-style malloc/free uprobes

This example mirrors the OpenTelemetry eBPF profiler's generic probe attach
path: a Go loader uses cilium/ebpf `link.OpenExecutable(...).Uprobe(...)` to
attach one generic kprobe-typed BPF program to libc `malloc` and `free`.

The Go runtime issues BPF-related syscalls directly, so bpftime needs the daemon
path for this loader. The LD_PRELOAD syscall-server path is not enough for the
tracer process itself, but the target process still uses the bpftime agent.

## Build

```sh
cmake --build build --target bpftime_daemon bpftime-agent -j$(nproc)
make -C example/otel-ebpf-profiler
make -C example/otel-ebpf-profiler tracer
```

## Run

Start the daemon from the repository root:

```sh
sudo build/daemon/bpftime_daemon -v
```

In another shell, start the cilium/ebpf loader:

```sh
cd example/otel-ebpf-profiler
sudo ./otel_malloc_free
```

From the repository root, run the victim with the bpftime agent preloaded:

```sh
sudo env LD_PRELOAD=$PWD/build/runtime/agent/libbpftime-agent.so \
  example/otel-ebpf-profiler/victim
```

The tracer should print per-pid `malloc` and `free` counts. `malloc` and `free`
are distinguished by `bpf_get_attach_cookie()`, which exercises the daemon's
`BPF_LINK_CREATE` handling for cilium/ebpf-created perf-event links.
