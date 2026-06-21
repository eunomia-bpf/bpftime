# OpenTelemetry eBPF profiler malloc/free uprobes

This example has two paths:

- **Full OpenTelemetry profiler path**: fetch and run the upstream
  `opentelemetry-ebpf-profiler` collector receiver, configured with
  `probe_links` for libc `malloc` and `free`.
- **Minimal cilium/ebpf smoke path**: build the small local loader in this
  directory to exercise the same cilium `link.OpenExecutable(...).Uprobe(...)`
  attach shape and bpftime perf-event link cookie handling.

The full path is the primary one. bpftime's daemon handles BPF syscalls from the
OTel profiler process, while the target process runs with `libbpftime-agent.so`
preloaded.

## Build bpftime and the local workload

From the bpftime repository root:

```sh
cmake --build build --target bpftime_daemon bpftime-agent -j$(nproc)
make -C example/otel-ebpf-profiler
make -C example/otel-ebpf-profiler tracer
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

## Run the full OTel profiler on bpftime

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

## How the full OTel path attaches malloc/free

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

The local minimal loader only counts calls. The full OTel profiler collects
stacks. Its generic probe enters `support/ebpf/generic_probe.ebpf.c`, calls
`collect_trace(ctx, TRACE_PROBE, pid, tid, ts, 0)`, and then the normal OTel
unwinding pipeline runs:

- collect kernel frames,
- derive user registers from `pt_regs`,
- look up process and mapping metadata,
- tail-call the native and interpreter unwinders,
- report profiles through the OTel profiles pipeline.

That is why the full path uses the upstream OTel binary instead of copying only
one BPF program into this example.

## Minimal cilium/ebpf smoke test

This path is smaller and easier to debug when only validating bpftime attach
semantics:

```sh
sudo build/daemon/bpftime_daemon -v
```

In another shell:

```sh
cd example/otel-ebpf-profiler
sudo ./otel_malloc_free
```

Then run the target from the repository root:

```sh
sudo env LD_PRELOAD=$PWD/build/runtime/agent/libbpftime-agent.so \
  example/otel-ebpf-profiler/victim
```

The local tracer prints per-pid `malloc` and `free` counts. It distinguishes
the two probes with `bpf_get_attach_cookie()`, which is a focused regression
test for bpftime's cilium perf-event link cookie support.

## Performance comparison

For a quick local comparison:

```sh
ITERATIONS=200000 REPEATS=5 \
  example/otel-ebpf-profiler/scripts/benchmark.sh
```

The benchmark writes CSV results under `.run-logs/` and reports average wall
time for:

- `baseline`: victim under `sudo`, no bpftime agent,
- `bpftime-agent`: victim with the bpftime agent connected to an empty daemon,
- `minimal-cilium-loader`: victim with the local malloc/free cilium loader
  attached through bpftime.

To include the full upstream OTel collector profiler in the benchmark:

```sh
RUN_FULL_OTEL=1 ITERATIONS=200000 REPEATS=3 \
  example/otel-ebpf-profiler/scripts/benchmark.sh
```

Use longer runs for stable numbers. The full OTel path measures the real stack
collection and reporting pipeline, while the minimal loader isolates bpftime's
uprobe attach and execution overhead.
