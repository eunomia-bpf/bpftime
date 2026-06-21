# OpenTelemetry eBPF profiler malloc/free uprobes

This example has two paths plus a native-kernel comparison mode:

- **OpenTelemetry profiler daemon mirror path**: fetch and run the upstream
  `opentelemetry-ebpf-profiler` collector receiver, configured with
  `probe_links` for libc `malloc` and `free`, while bpftime mirrors the
  kernel-created handlers into the target process.
- **Minimal cilium/ebpf smoke path**: build the small local loader in this
  directory to exercise the same cilium `link.OpenExecutable(...).Uprobe(...)`
  attach shape and bpftime perf-event link cookie handling.
- **Native-kernel comparison path**: run the same minimal loader or upstream
  OTel collector without bpftime to get a kernel reference.

The upstream OTel binaries are statically linked Go programs that issue raw
syscalls, so `LD_PRELOAD=libbpftime-syscall-server.so` does not intercept their
BPF syscalls. This example therefore uses daemon mirror mode for upstream OTel:
the OTel collector still attaches through the kernel, and bpftime mirrors those
handlers for an agent-preloaded target process. The collector's exported profiles
come from the upstream OTel kernel pipeline.

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

## Run the OTel profiler daemon mirror path

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
including cilium perf-event backed uprobes and BPF link cookies. It is not a
pure syscall-server run of the static Go OTel collector.

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

The local minimal loader only counts calls. The upstream OTel profiler collects
stacks in its native kernel pipeline. Its generic probe enters
`support/ebpf/generic_probe.ebpf.c`, calls
`collect_trace(ctx, TRACE_PROBE, pid, tid, ts, 0)`, and then the normal OTel
unwinding pipeline runs:

- collect kernel frames,
- derive user registers from `pt_regs`,
- look up process and mapping metadata,
- tail-call the native and interpreter unwinders,
- report profiles through the OTel profiles pipeline.

That is why the daemon mirror path uses the upstream OTel binary instead of
copying only one BPF program into this example. The current upstream static Go
collector does not consume trace ringbuf data produced by bpftime's userspace
execution path.

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
- `daemon-mirror-minimal-cilium-loader`: local malloc/free cilium loader attached
  through the kernel and mirrored into bpftime.

To add a native-kernel minimal-loader reference:

```sh
RUN_KERNEL_COMPARE=1 ITERATIONS=200000 REPEATS=3 \
  example/otel-ebpf-profiler/scripts/benchmark.sh
```

This adds:

- `kernel-minimal-cilium-loader`: same local loader and victim without bpftime.

To include the full upstream OTel collector profiler in the benchmark:

```sh
RUN_FULL_OTEL=1 ITERATIONS=200000 REPEATS=3 \
  example/otel-ebpf-profiler/scripts/benchmark.sh
```

With both flags enabled, the benchmark also includes:

- `daemon-mirror-full-otel-collector`: upstream OTel collector attached through
  the kernel while bpftime mirrors handlers into the target process,
- `kernel-full-otel-collector`: upstream OTel collector without bpftime.

Use longer runs for stable numbers. Do not interpret the daemon mirror full-OTel
case as pure bpftime stack-collection overhead; the upstream static Go collector
cannot be intercepted by `LD_PRELOAD`, and its exported profiles come from the
kernel OTel pipeline. The minimal daemon mirror case is a focused regression for
bpftime uprobe attach/execution semantics, while the `kernel-*` cases provide
native-kernel references.
