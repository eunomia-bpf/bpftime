# BB Tracepoint Register Capture Example

This example shows a minimal setup for basic-block tracepoints with register capture on a CUDA kernel.

It demonstrates:

- Using a BB attach point: `kprobe/bb_reg_kernel__BB0__r2__r5`
- Capturing PTX registers via `bpf_get_ptx_reg()`
- Reading captured values from a BPF map in userspace
- A simple `0xdeadbeef` data path in kernel code to make values easy to reason about

## Files

- `bb_register_kernel.cu`: CUDA program with `bb_reg_kernel`
- `bb_register_kernel_asm.cu`: CUDA program with `bb_reg_kernel`, but with pre-compiled asms
- `bb_register_probe.bpf.c`: eBPF BB tracepoint program
- `bb_register_probe.c`: userspace loader and map printer
- `Makefile`: build rules for both probe and CUDA app

## Kernel Logic

The CUDA kernel computes:

1. `marker = 0xdeadbeef`
2. `marker ^= threadIdx.x`
3. If `threadIdx.x` is even, `marker += 1`

For lane 0 (`threadIdx.x == 0`), expected value is `0xdeadbef0`.

## Build

From the repository root:

```bash
make -C example/gpu/bb-register
```

## Run

Start the eBPF server process first:

```bash
BPFTIME_LOG_OUTPUT=console LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  example/gpu/bb-register/bb_register_probe
```

In another terminal, run the CUDA client process:

```bash
BPFTIME_LOG_OUTPUT=console LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  example/gpu/bb-register/bb_register_kernel_asm
```

## Expected Output

The loader prints sampled register values from BB0 every second, for lane 0 only.

Typical output looks like:

```text
hits=12 r2=0xdeadbeef r5=0xdeadbef0 expected_r2(tid0)=0xdeadbeef expected_r5(tid0)=0xdeadbef0
```

`r2` and `r5` are captured in the order declared in the section suffix (`__r2__r5`).

## Notes

- Register allocation can vary with compiler flags and CUDA versions. If your values differ, inspect generated PTX and adjust the captured register names, or try the asm version first
- This demo needs runtime support for helper `512` (`bpf_get_ptx_reg`). If logs show `Ext func not found: _bpf_helper_ext_0512`, rebuild bpftime/NV attach with BB register helper support enabled.
- This example is intentionally small and focused on BB/register mechanics; it is a good starting point for larger tracepoint programs.
