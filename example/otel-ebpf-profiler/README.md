# OpenTelemetry eBPF profiler malloc/free uprobes

This example runs the upstream `opentelemetry-ebpf-profiler` collector receiver
with `probe_links` for libc `malloc` and `free`. It uses the complete upstream
OTel eBPF profiler stack collection pipeline.

The upstream OTel binaries are statically linked Go programs that issue raw
syscalls, so `LD_PRELOAD=libbpftime-syscall-server.so` does not intercept their
BPF syscalls. This example therefore uses daemon mirror compatibility mode:
the OTel collector attaches through the kernel, bpftime mirrors those handlers
for an agent-preloaded target process, and the collector's exported profiles
come from the native kernel OTel pipeline.

## Build bpftime and the workload

From the bpftime repository root:

```sh
cmake --build build --target bpftime_daemon bpftime-agent -j$(nproc)
make -C example/otel-ebpf-profiler
```

## Build the upstream OTel profiler

The OTel source tree is cloned under
`example/otel-ebpf-profiler/.otel-ebpf-profiler` by default and is ignored by
git:

```sh
example/otel-ebpf-profiler/scripts/build-upstream-otel.sh
```

Useful overrides:

```sh
OTEL_EBPF_PROFILER_DIR=/tmp/opentelemetry-ebpf-profiler \
OTEL_EBPF_PROFILER_REF=main \
OTEL_EBPF_PROFILER_BUILD_TARGETS="otelcol-ebpf-profiler ebpf-profiler" \
  example/otel-ebpf-profiler/scripts/build-upstream-otel.sh
```

The upstream project currently requires Go 1.25 for native builds. The script
sets `GOTOOLCHAIN=auto` unless you override it.

## Run the full OTel profiler

After building `otelcol-ebpf-profiler`, run:

```sh
RUN_SECONDS=20 example/otel-ebpf-profiler/scripts/run-full-otel-profiler.sh
```

The script:

1. finds libc,
2. starts `build/daemon/bpftime_daemon`,
3. starts the upstream `otelcol-ebpf-profiler` with
   `config/otelcol-malloc-free.yaml`,
4. runs the local `victim` with `libbpftime-agent.so` preloaded,
5. writes collector, daemon, and workload logs under `.run-logs/`.

This validates bpftime compatibility with the upstream OTel loader shape,
including cilium perf-event backed uprobes, BPF link cookies, and OTel's
stack-collection maps and tail-call unwinder programs. It is not a pure
syscall-server run of the static Go OTel collector.

The script defaults `BPFTIME_MAX_FD_COUNT=65536` for the daemon because daemon
mirror mode reuses kernel object IDs as bpftime shared-memory IDs, and those IDs
can exceed bpftime's smaller default on long-running hosts. Override it in the
environment if you need a different table size.

The collector config uses the real OTel profiling receiver:

```yaml
receivers:
  profiling:
    probe_links:
      - ${env:OTEL_EBPF_PROFILER_MALLOC_PROBE}
      - ${env:OTEL_EBPF_PROFILER_FREE_PROBE}
```

The script sets those variables to:

```sh
uprobe:/path/to/libc.so.6:malloc
uprobe:/path/to/libc.so.6:free
```

If you already have an upstream checkout or binary, point the script at it:

```sh
OTEL_EBPF_PROFILER_DIR=/tmp/opentelemetry-ebpf-profiler \
OTEL_COLLECTOR_BIN=/tmp/opentelemetry-ebpf-profiler/otelcol-ebpf-profiler \
  example/otel-ebpf-profiler/scripts/run-full-otel-profiler.sh
```

## How the OTel profiler attaches malloc/free

Upstream OTel parses `probe_links` in `tracer/probe.go`. For a user probe such
as `uprobe:/lib/x86_64-linux-gnu/libc.so.6:malloc`, it opens the target ELF via
cilium/ebpf and attaches the generic program:

```go
ex, err := link.OpenExecutable(spec.Target)
return ex.Uprobe(spec.Symbol, prog, nil)
```

The generic program is named `kprobe__generic`. bpftime must therefore support
cilium's `BPF_LINK_CREATE` path for perf-event backed uprobes.

## How stacks are collected

The upstream OTel profiler collects stacks through its normal eBPF pipeline.
Its generic probe enters `support/ebpf/generic_probe.ebpf.c`, calls
`collect_trace(ctx, TRACE_PROBE, pid, tid, ts, 0)`, and then the normal OTel
unwinding pipeline runs:

- collect kernel frames,
- derive user registers from `pt_regs`,
- look up process and mapping metadata,
- tail-call the native and interpreter unwinders,
- report profiles through the OTel profiles pipeline.

That is why this example uses the upstream OTel binary instead of a local BPF
counter. The current upstream static Go collector does not consume
trace ringbuf data produced by bpftime's userspace execution path.

## Benchmark full OTel stack collection

The benchmark runs only the complete upstream OTel profiler. It does not use a
local BPF loader.

```sh
OTEL_EBPF_PROFILER_DIR=/tmp/opentelemetry-ebpf-profiler \
RUN_SECONDS=10 REPEATS=3 \
  example/otel-ebpf-profiler/scripts/benchmark.sh
```

The benchmark writes CSV results under `.run-logs/` and reports elapsed time,
completed malloc/free iterations, and iterations per second for:

- `baseline`: victim under `sudo`, no profiler,
- `otel-kernel-stack-collector`: upstream OTel profiler attached through the
  kernel, collecting stacks for the victim,
- `otel-daemon-mirror-stack-collector`: upstream OTel profiler attached through
  the kernel while bpftime mirrors handlers into the agent-preloaded victim.

Each profiler run waits for the OTel debug exporter and fails if the collector
log does not contain exported profile data for the `victim` process. The timed
region is the victim workload only; collector startup and flush time are kept
outside the measured interval.
