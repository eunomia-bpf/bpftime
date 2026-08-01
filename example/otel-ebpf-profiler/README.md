# OpenTelemetry eBPF profiler compatibility example

This example runs the upstream OpenTelemetry eBPF profiler with `malloc` and
`free` uprobes while the bpftime daemon mirrors the cilium/ebpf perf-event
links. It checks the link shape supported by bpftime, including the
`BPF_LINK_CREATE` path used by the profiler.

This is intentionally a daemon-mirror compatibility example. The upstream Go
collector loads and runs its eBPF programs in the kernel, and the exported
profiles come from that kernel pipeline. The full profiler program is not run
by the bpftime agent because its map and helper requirements are not all
supported yet.

## Prerequisites

Build the daemon from the repository root:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --target bpftime_daemon -j"$(nproc)"
```

Build the tested upstream profiler revision:

```sh
git clone https://github.com/open-telemetry/opentelemetry-ebpf-profiler.git
cd opentelemetry-ebpf-profiler
git checkout 39d09634147c707f035ddda78186054bfe037629
make ebpf otelcol-ebpf-profiler
```

The upstream build currently requires Go 1.25 and LLVM 17.

## Run

From this directory, point the example at the upstream binary:

```sh
make
sudo env \
  OTEL_COLLECTOR_BIN=/path/to/opentelemetry-ebpf-profiler/otelcol-ebpf-profiler \
  ./scripts/run.sh
```

If bpftime uses a non-default build directory, also set `BPFTIME_DAEMON`:

```sh
sudo env \
  BPFTIME_DAEMON=/path/to/build/daemon/bpftime_daemon \
  OTEL_COLLECTOR_BIN=/path/to/otelcol-ebpf-profiler \
  ./scripts/run.sh
```

The script succeeds only after the daemon observes both profiler uprobes and
the collector exports a profile containing the example victim. Logs are kept
under `.run-logs/`.
