# kernel_trace - CUDA kernel trace demo

`kernel_trace` is a minimal end-to-end example showing how bpftime can instrument a CUDA kernel. It injects an eBPF program into the `vectorAdd` kernel in `vec_add.cu`, collects per-thread block/thread indices and the GPU global timer, and sends events back to the host through the CUDA trampoline.

This demo also showcases the **stub-based attach** pattern: the CUDA kernel contains a nearly empty hook stub
`__bpftime_cuda__kernel_trace()` and calls it once per thread. At PTX time, bpftime rewrites that call so it
targets the eBPF-generated probe function instead of the dummy stub.

## Build

```bash
# from repo root
cmake -B build -DBPFTIME_ENABLE_CUDA_ATTACH=1 -DBPFTIME_CUDA_ROOT=/usr/local/cuda
cmake --build build

# build the demo (BPF, userspace loader, sample CUDA program)
make -C example/gpu/kernel_trace
```

## Run

The demo needs the syscall server (for map polling) and the agent (to intercept CUDA fatbins). Use two terminals:

### Terminal 1 – start the tracer

```bash
BPFTIME_MAP_GPU_THREAD_COUNT=8192 \
BPFTIME_SHM_MEMORY_MB=256 \
BPFTIME_LOG_OUTPUT=console \
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
example/gpu/kernel_trace/kernel_trace 
```

### Terminal 2 – launch the sample CUDA program

```bash
BPFTIME_LOG_OUTPUT=console \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
example/gpu/kernel_trace/vec_add
```

You should see lines like:

```text
[kernel_trace] ts=809635273301344 block=(0,0,0) thread=(0,0,0)
[kernel_trace] ts=809635273301456 block=(0,0,0) thread=(1,0,0)
[kernel_trace] total events: 256
```

Each line corresponds to one GPU thread entering `vectorAdd`, confirming that the hook is active.

## Customizing the hook

- To trace a different kernel, change the section name in `kernel_trace.bpf.c`:

  ```c
  SEC("kprobe/_Z9vectorAddPKfS0_Pf")
  ```

  Use `cuobjdump -symbols your_app | grep <kernel>` to find other C++ mangled kernel names.

- To capture more fields (arguments, computed values, etc.), extend `struct kernel_trace_event` and keep the struct definition in `kernel_trace.bpf.c` and `kernel_trace.c` in sync.
- You can also replace `vectorAdd` with your own CUDA program, as long as you preload the agent so the fatbin interception and PTX rewriting pipeline are active.

## How it works

1. The bpftime agent intercepts `vectorAdd`’s fatbin registration, rewrites PTX, and injects the trampoline.
2. Inside `vec_add.cu`, the kernel calls a dummy device function:

   ```c++
   __device__ __noinline__ void __bpftime_cuda__kernel_trace() {}

   __global__ void vectorAdd(...) {
       ...
       __bpftime_cuda__kernel_trace(); // hook point
       ...
   }
   ```

   The PTX pass for `kprobe` entry sees the `call __bpftime_cuda__kernel_trace` instruction and redirects it to
   the eBPF-generated PTX function for this kernel. If no such stub call exists, the pass falls back to injecting
   a call at the kernel entry, so existing code keeps working.

3. The eBPF program uses GPU helper IDs (`503`, `505`, `502`) to collect thread coordinates and timestamps.
4. Events are written to a GPU ring buffer map (`BPF_MAP_TYPE_GPU_RINGBUF_MAP`), which the syscall server exposes to userspace.
5. The userspace loader polls that map and prints each event.

This serves as a minimal template for hooking CUDA kernels and running custom eBPF logic on the GPU, which you can extend into more advanced tracepoints or profilers.
