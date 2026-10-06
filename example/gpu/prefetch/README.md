# GPU prefetch example

This example demonstrates a bpftime GPU kprobe that issues PTX
`prefetch.global.L2` instructions before a CUDA UVM streaming kernel touches
its next pages.

The BPF program attaches to two points:

- `uprobe/./prefetch_example:launch_run_seq_kernel` captures the current
  `RunSeqConfig`.
- `kprobe/_Z16seq_chunk_kernelPKfPfmmimi` runs on the GPU and prefetches input
  and output pages for each CUDA thread before `seq_chunk_kernel` accesses them.

The prefetch helper is registered as GPU helper ID 512. Helper IDs 509, 510,
and 511 are reserved for SM, warp, and lane ID helpers.

## Build

```sh
make -C example/gpu/prefetch
```

This builds:

- `prefetch`: the bpftime loader program for `prefetch.bpf.c`.
- `prefetch_example`: synthetic UVM streaming kernels.
- `prefetch_gemm`: GEMM-oriented UVM workload.

## Run the bpftime prefetch probe

Start the BPF program in one terminal:

```sh
cd example/gpu/prefetch
bpftime load ./prefetch
```

Run the workload in another terminal:

```sh
cd example/gpu/prefetch
bpftime start ./prefetch_example --kernel=seq_stream --mode=uvm --size_factor=1.5 --stride_bytes=4096 --iterations=5
```

The `seq_stream` workload launches `seq_chunk_kernel`; the GPU kprobe performs
the prefetch work. For comparison without bpftime, run the executable directly:

```sh
./prefetch_example --kernel=seq_stream --mode=uvm --size_factor=1.5 --stride_bytes=4096 --iterations=5
```

## Inline CUDA prefetch baseline

The synthetic benchmark also includes an inline CUDA baseline that emits
`prefetch.global.L2` from the workload kernel itself:

```sh
./prefetch_example --kernel=seq_uvm_prefetch --mode=uvm --size_factor=1.5 --stride_bytes=4096 --iterations=5
```

This baseline does not use bpftime; it is useful for comparing the kprobe-based
prefetch path with a native CUDA implementation.

## GEMM workload

`prefetch_gemm` is a native CUDA comparison workload; its GEMM kernel emits
inline `prefetch.global.L2` instructions and does not use the bpftime
`seq_chunk_kernel` kprobe above.

```sh
./prefetch_gemm --kernel=gemm --mode=uvm --size_factor=1.5 --stride_bytes=4096 --iterations=5
```
