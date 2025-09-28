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
  - [Error injections](#error-injections)
  - [Nginx eBPF module](#nginx-ebpf-module)
  - [Use DPDK with userspace eBPF to run XDP seamlessly](#use-dpdk-with-userspace-ebpf-to-run-xdp-seamlessly)
  - [Use the vm only(No runtime, No uprobe) as a library](#use-the-vm-onlyno-runtime-no-uprobe-as-a-library)

## minimal examples

See [example/minimal](https://github.com/eunomia-bpf/bpftime/tree/master/example/minimal) and [usdt_minimal](https://github.com/eunomia-bpf/bpftime/tree/master/example/usdt_minimal).

The bpftime supports the following types of eBPF programs:

- `uprobe/uretprobe`: trace userspace functions at start or and.
- `syscall tracepoints`: trace the specific syscall types.
- `USDT`: trace the userspace functions with USDT.

You may use `bpf_override_return` to change the control flow of the program.

See [documents/available-features.md](https://github.com/eunomia-bpf/bpftime/tree/master/documents/avaliable-features.md) for more details.

## Tracing the system

### Tracing userspace functions with uprobe

Attach uprobe, uretprobe or all syscall tracepoints(currently x86 only) eBPF programs to a process or a group of processes

- [`malloc`](https://github.com/eunomia-bpf/bpftime/tree/master/example/malloc): count the malloc calls in libc by pid. demonstrate how to use the userspace `uprobe` with basic `hashmap`.
- [`bashreadline`](https://github.com/eunomia-bpf/bpftime/tree/master/example/libbpf-tools/bashreadline): Print entered bash commands from running shells,
- [`sslsniff`](https://github.com/eunomia-bpf/bpftime/tree/master/example/sslsniff): Trace and print all SSL/TLS connections and raw traffic data.

### tracing all syscalls with tracepoints

- [`opensnoop`](https://github.com/eunomia-bpf/bpftime/tree/master/example/opensnoop): trace file open or close syscalls in a process. demonstrate how to use the userspace `syscall tracepoint` with `ring buffer` output.

More bcc/libbpf-tools examples can be found in [example/libbpf-tools](https://github.com/eunomia-bpf/bpftime/tree/master/example/libbpf-tools).

### bpftrace

You can also run bpftime with `bpftrace`, we've test it on [this commit](https://github.com/iovisor/bpftrace/commit/75aca47dd8e1d642ff31c9d3ce330e0c616e5b96).

It should be able to work with the bpftrace from the package manager of your distribution, for example:

```bash
sudo apt install bpftrace
```

Or you can build the latest bpftrace from source.

More details about how to run bpftrace in usespace, can be found in [example/bpftrace](https://github.com/eunomia-bpf/bpftime/tree/master/example/bpftrace).

### Use bpftime to trace SPDK

See <https://github.com/eunomia-bpf/bpftime/wiki/Benchmark-of-SPDK> for how to use bpftime to trace SPDK.

## Error injections

- [`error-injection`](https://github.com/eunomia-bpf/bpftime/tree/master/example/error-inject) Inject errors into userspace functions or syscalls to test its error handling capabilities.

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
