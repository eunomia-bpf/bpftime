# OpenTelemetry eBPF profiler compatibility example

This example runs the unmodified upstream
[OpenTelemetry eBPF profiler](https://github.com/open-telemetry/opentelemetry-ebpf-profiler)
(a Go program built on `cilium/ebpf`) and drives its `malloc`/`free` uprobes
against a looping victim on top of bpftime. Two modes are provided:

- **`MODE=userspace` (default, rootless):** the profiler's eBPF programs, maps
  and links are emulated completely in userspace by bpftime. No `sudo`, no
  kernel eBPF, no `bpftime_daemon`.
- **`MODE=daemon` (legacy):** `bpftime_daemon` mirrors the cilium/ebpf
  perf-event links while the collector still runs in the kernel. Requires root.

## How the rootless mode works

The Go collector issues `bpf(2)` and `perf_event_open(2)` through raw syscall
instructions that bypass glibc. With `BPFTIME_GO_SECCOMP=1`, the bpftime
syscall-server installs a seccomp `USER_NOTIF` filter from a dynamic-loader
constructor (covering the Go runtime entry point) and emulates every captured
call in its fake kernel: synthetic `/proc/kallsyms` and tracefs event ids, the
profiler's map types (including `PROG_ARRAY` tail calls and `LPM_TRIE`), and
`perf_event_open` uprobe/uretprobe links with `bpf_cookie`. The collector and
the victim open the same named shared-memory segment
(`BPFTIME_GLOBAL_SHM_NAME`); the victim's `libbpftime-agent.so` constructor
enumerates the published handlers and instantiates the `malloc`/`free` uprobes
in-process with Frida. The profiler then receives real samples with the
victim's call stacks and exports them via its debug exporter.

## Prerequisites

Build the bpftime runtime libraries from the repository root (the LLVM JIT is
not required; the uBPF interpreter is enough):

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_BPFTIME_DAEMON=OFF
cmake --build build --target bpftime-agent bpftime-syscall-server -j"$(nproc)"
```

Build the tested upstream profiler revision from a `git clone`; Go 1.25 and
LLVM 17 are required:

```sh
git clone https://github.com/open-telemetry/opentelemetry-ebpf-profiler.git
cd opentelemetry-ebpf-profiler
git checkout 39d09634147c707f035ddda78186054bfe037629
make ebpf otelcol-ebpf-profiler
```

### Rootless mode requires a dynamic collector

The default `make` target sets `CGO_ENABLED=0` and produces a statically
linked executable with no `PT_INTERP`, so `LD_PRELOAD` (and the bpftime
seccomp constructor) never runs. Rebuild the collector as a dynamic ELF —
build flags only, no upstream source change:

```sh
cd cmd/otelcol-ebpf-profiler
CGO_ENABLED=1 go build -tags "osusergo,netgo" \
  -ldflags "-linkmode=external" -o ../../otelcol-ebpf-profiler-dyn
```

Verify it is dynamic with `readelf -l otelcol-ebpf-profiler-dyn | grep INTERP`
and point `OTEL_COLLECTOR_BIN` at this `*-dyn` binary. The legacy
`MODE=daemon` path accepts either binary because it does not preload the
collector.

## Run (rootless userspace, no sudo)

From this directory, point the example at the dynamic collector. The script
auto-detects the victim's `libc.so.6` via `ldd`, so the same command works on
`x86_64` and `aarch64` (override with `LIBC_PATH`):

```sh
cc -O2 -g -Wall -Wextra victim-src/victim.c -o victim
OTEL_COLLECTOR_BIN=/path/to/opentelemetry-ebpf-profiler/otelcol-ebpf-profiler-dyn \
  ./scripts/run.sh
```

Useful overrides: `BUILD_DIR` (or `BPFTIME_REPO_ROOT`), `RUN_SECONDS`,
`LOG_DIR`, `LIBC_PATH`, `BPFTIME_SHM_MEMORY_MB`; the shared-memory name is
generated uniquely per run. The script fails unless all of the following are
observed:

1. bpftime creates both `malloc` and `free` userspace uprobe perf-event
   handlers on the detected `libc.so.6`;
2. both profiler `BPF_LINK_CREATE` perf-event links (with `bpf_cookie`) are
   emulated and attached in userspace;
3. the bpftime agent reports `Attach successfully` inside the victim;
4. the real collector exports a `Profiles` payload whose
   `process.executable.name` is `victim`.

Logs under `.run-logs/<timestamp>/`: `collector.log` (collector output with
exported profiles), `bpftime-collector.log` and `bpftime-victim.log`
(diagnostics), and `victim.log` (the victim's own output).

## Run (legacy daemon kernel-mirror)

The legacy path keeps the profiler in the kernel and only mirrors its links
through `bpftime_daemon`, so it needs root and a daemon build:

```sh
cmake --build build --target bpftime_daemon -j"$(nproc)"
sudo env \
  OTEL_COLLECTOR_BIN=/path/to/opentelemetry-ebpf-profiler/otelcol-ebpf-profiler \
  MODE=daemon ./scripts/run.sh
```
