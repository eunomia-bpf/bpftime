# CUDA Graph eBPF Probe Example

This example shows how `bpftime` instruments kernels executed through CUDA Graphs.

- The CUDA app `vec_add_graph` builds a CUDA Graph containing a `vectorAdd` kernel and repeatedly launches it with `cudaGraphLaunch`.
- The eBPF program `cuda_probe` attaches to the `vectorAdd` kernel entry/exit points, even when the kernel is launched from a graph.

## Build

From the `bpftime/` repository root, build bpftime with CUDA attach enabled:

```bash
cmake -Bbuild -DBPFTIME_ENABLE_CUDA_ATTACH=1 -DBPFTIME_CUDA_ROOT=/usr/local/cuda .
cmake --build build -j"$(nproc)"
```

Build the example:

```bash
# or: export PATH=$BPFTIME_CUDA_ROOT/bin:$PATH
make -C example/gpu/cudagraph
```

## Run

Open two terminals.

**Terminal 1 (server, loads eBPF program):**

```bash
BPFTIME_LOG_OUTPUT=console BPFTIME_GLOBAL_SHM_NAME=bpftime_maps_shm_graph \
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  example/gpu/cudagraph/cuda_probe
```

**Terminal 2 (client, runs CUDA Graph app):**

```bash
BPFTIME_LOG_OUTPUT=console BPFTIME_GLOBAL_SHM_NAME=bpftime_maps_shm_graph \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  example/gpu/cudagraph/vec_add_graph
```
This will print the eBPF output from `cuda_probe` confirming the graph-launched kernel is being traced.
