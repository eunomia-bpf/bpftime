# Examples & Use Cases

## Table of Contents

- [Examples \& Use Cases](#examples--use-cases)
  - [Table of Contents](#table-of-contents)
  - [minimal examples](#minimal-examples)
  - [Tracing the system](#tracing-the-system)
    - [Tracing userspace functions with uprobe](#tracing-userspace-functions-with-uprobe)
    - [tracing all syscalls with tracepoints](#tracing-all-syscalls-with-tracepoints)
    - [bpftrace](#bpftrace)
    - [Use bpftime to trace SPDK](#use-bpftime-to-trace-spdk)
  - [BPF Features Demos](#bpf-features-demos)
  - [GPU/CUDA/ROCm Tracing](#gpucudarocm-tracing)
  - [Hotpatching Applications](#hotpatching-applications)
  - [Error injections](#error-injections)
  - [XDP in Userspace](#xdp-in-userspace)
  - [Advanced Examples](#advanced-examples)
    - [Attach Implementation](#attach-implementation)
    - [Using bpftime as a Library](#using-bpftime-as-a-library)
  - [Nginx eBPF module](#nginx-ebpf-module)
  - [Use DPDK with userspace eBPF to run XDP seamlessly](#use-dpdk-with-userspace-ebpf-to-run-xdp-seamlessly)
  - [Use the vm only(No runtime, No uprobe) as a library](#use-the-vm-onlyno-runtime-no-uprobe-as-a-library)

## minimal examples

See [example/minimal](https://github.com/eunomia-bpf/bpftime/tree/master/example/minimal) for basic examples demonstrating core bpftime features:

- [`uprobe`](https://github.com/eunomia-bpf/bpftime/tree/master/example/minimal): Basic uprobe example
- [`syscall`](https://github.com/eunomia-bpf/bpftime/tree/master/example/minimal): Syscall tracing example
- [`uprobe-override`](https://github.com/eunomia-bpf/bpftime/tree/master/example/minimal): Demonstrates using `bpf_override_return` to change function return values
- [`usdt_minimal`](https://github.com/eunomia-bpf/bpftime/tree/master/example/minimal/usdt_minimal): User Statically Defined Tracing (USDT) example

The bpftime supports the following types of eBPF programs:

- `uprobe/uretprobe`: trace userspace functions at start or end.
- `syscall tracepoints`: trace the specific syscall types.
- `USDT`: trace the userspace functions with USDT.

You may use `bpf_override_return` to change the control flow of the program.

See [documents/available-features.md](https://github.com/eunomia-bpf/bpftime/tree/master/documents/avaliable-features.md) for more details.

## Tracing the system

### Tracing userspace functions with uprobe

Attach uprobe, uretprobe or all syscall tracepoints(currently x86 only) eBPF programs to a process or a group of processes

- [`malloc`](https://github.com/eunomia-bpf/bpftime/tree/master/example/malloc): count the malloc calls in libc by pid. demonstrate how to use the userspace `uprobe` with basic `hashmap`.
- [`bashreadline`](https://github.com/eunomia-bpf/bpftime/tree/master/example/tracing/bashreadline): Print entered bash commands from running shells.
- [`sslsniff`](https://github.com/eunomia-bpf/bpftime/tree/master/example/tracing/sslsniff): Trace and print all SSL/TLS connections and raw traffic data.
- [`funclatency`](https://github.com/eunomia-bpf/bpftime/tree/master/example/tracing/funclatency): Measure function latency distribution.
- [`goroutine`](https://github.com/eunomia-bpf/bpftime/tree/master/example/tracing/goroutine): Trace Go runtime goroutine operations.

### tracing all syscalls with tracepoints

- [`opensnoop`](https://github.com/eunomia-bpf/bpftime/tree/master/example/tracing/opensnoop): trace file open or close syscalls in a process. demonstrate how to use the userspace `syscall tracepoint` with `ring buffer` output.
- [`opensnoop_ring_buf`](https://github.com/eunomia-bpf/bpftime/tree/master/example/tracing/opensnoop_ring_buf): Alternative implementation using ring buffer for output.
- [`statsnoop`](https://github.com/eunomia-bpf/bpftime/tree/master/example/tracing/statsnoop): Trace stat() syscalls system-wide.
- [`syscount`](https://github.com/eunomia-bpf/bpftime/tree/master/example/tracing/syscount): Count system calls by type and process.
- [`mountsnoop`](https://github.com/eunomia-bpf/bpftime/tree/master/example/tracing/mountsnoop): Trace mount and umount system calls.
- [`sigsnoop`](https://github.com/eunomia-bpf/bpftime/tree/master/example/tracing/sigsnoop): Trace signals sent to processes.

More bcc/libbpf-tools style examples can be found in [example/tracing](https://github.com/eunomia-bpf/bpftime/tree/master/example/tracing).

### bpftrace

You can also run bpftime with `bpftrace`, we've test it on [this commit](https://github.com/iovisor/bpftrace/commit/75aca47dd8e1d642ff31c9d3ce330e0c616e5b96).

It should be able to work with the bpftrace from the package manager of your distribution, for example:

```bash
sudo apt install bpftrace
```

Or you can build the latest bpftrace from source.

More details about how to run bpftrace in userspace, can be found in [example/tracing/bpftrace](https://github.com/eunomia-bpf/bpftime/tree/master/example/tracing/bpftrace).

### Use bpftime to trace SPDK

See <https://github.com/eunomia-bpf/bpftime/wiki/Benchmark-of-SPDK> for how to use bpftime to trace SPDK.

## BPF Features Demos

The [example/bpf_features](https://github.com/eunomia-bpf/bpftime/tree/master/example/bpf_features) directory contains demonstrations of various BPF map types and features:

- [`bloom_filter_demo`](https://github.com/eunomia-bpf/bpftime/tree/master/example/bpf_features/bloom_filter_demo): Demonstrates the use of BPF bloom filter maps for efficient set membership testing.
- [`get_stack_id_example`](https://github.com/eunomia-bpf/bpftime/tree/master/example/bpf_features/get_stack_id_example): Shows how to capture and use stack traces with `bpf_get_stackid`.
- [`lpm_trie_demo`](https://github.com/eunomia-bpf/bpftime/tree/master/example/bpf_features/lpm_trie_demo): Demonstrates Longest Prefix Match (LPM) trie maps for IP address matching.
- [`queue_demo`](https://github.com/eunomia-bpf/bpftime/tree/master/example/bpf_features/queue_demo): Examples of using BPF queue and stack maps for FIFO/LIFO data structures.
- [`tailcall_minimal`](https://github.com/eunomia-bpf/bpftime/tree/master/example/bpf_features/tailcall_minimal): Minimal example of BPF tail calls for program chaining.

## GPU/CUDA/ROCm Tracing

The [example/gpu](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu) directory contains examples for tracing GPU kernels:

- [`cuda-counter`](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/cuda-counter): Basic CUDA kernel tracing example
- [`cuda-counter-gpu-array`](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/cuda-counter-gpu-array): CUDA tracing with GPU array maps
- [`cuda-counter-gpu-ringbuf`](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/cuda-counter-gpu-ringbuf): CUDA tracing with GPU ring buffer
- [`rocm-counter`](https://github.com/eunomia-bpf/bpftime/tree/master/example/gpu/rocm-counter): AMD ROCm GPU kernel tracing

## Hotpatching Applications

The [example/hotpatch](https://github.com/eunomia-bpf/bpftime/tree/master/example/hotpatch) directory shows how to dynamically modify application behavior:

- [`redis`](https://github.com/eunomia-bpf/bpftime/tree/master/example/hotpatch/redis): Hotpatch Redis server behavior without modifying source code
- [`vim`](https://github.com/eunomia-bpf/bpftime/tree/master/example/hotpatch/vim): Example of hotpatching Vim editor

## Error injections

- [`error-injection`](https://github.com/eunomia-bpf/bpftime/tree/master/example/error-inject): Inject errors into userspace functions or syscalls to test its error handling capabilities.

## XDP in Userspace

- [`xdp-counter`](https://github.com/eunomia-bpf/bpftime/tree/master/example/xdp-counter): Example of running XDP programs in userspace for packet processing

## Advanced Examples

### Attach Implementation

The [example/attach_implementation](https://github.com/eunomia-bpf/bpftime/tree/master/example/attach_implementation) directory contains a complete example of implementing a high-performance nginx request filter using bpftime, including benchmarks comparing different filtering approaches (eBPF, WASM, LuaJIT, RLBox, etc.).

### Using bpftime as a Library

- [`libbpftime_example`](https://github.com/eunomia-bpf/bpftime/tree/master/example/libbpftime_example): Example of using bpftime's shared memory and runtime features as a library

## Nginx eBPF module

A nginx eBPF module is implemented with bpftime, which can be used to extend nginx with eBPF programs.

See <https://github.com/eunomia-bpf/Nginx-eBPF-module>

## Use DPDK with userspace eBPF to run XDP seamlessly

See <https://github.com/eunomia-bpf/XDP-eBPF-in-DPDK>

We can run the same eBPF XDP program in both kernel and userspace, and the userspace eBPF program can be used to run XDP programs seamlessly. Unlike ubpf in DPDK, we don't need to modify the eBPF program, and can support eBPF maps

## Use the vm only(No runtime, No uprobe) as a library

The LLVM JIT or AOT can be used as a library, without the runtime and uprobe.

See the examples:

1. Cli: <https://github.com/eunomia-bpf/bpftime/tree/master/vm/cli>
2. Simple example: <https://github.com/eunomia-bpf/bpftime/tree/master/vm/example>
