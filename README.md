# bpftime: Userspace eBPF runtime for fast Uprobe & Syscall Tracing

Introducing `bpftime`, a full-featured, high-performance eBPF runtime designed to operate in userspace. It's engineered to offer fast Uprobe and Syscall tracing capabilities. Userspace uprobe can be **10x faster than kernel uprobe!**

This makes `bpftime` an ideal choice for use in embedded systems, IoT, edge computing, smart contracts, and cloud-native solutions. It's also compatible with existing eBPF toolchains like clang and libbpf, without requiring any modifications.

> ⚠️ **Note**: `bpftime` is actively under development. The API or design might change in upcoming releases, and it's not yet recommended for production use. See our [roadmap](#roadmap) for details.

## Key Features

- **Unprivileged Userspace eBPF**: Run eBPF programs in userspace, attaching them to Uprobes and Syscall tracepoints just as you would in the kernel.
- **Performance**: Experience up to a 10x speedup in Uprobe overhead compared to kernel uprobe and uretprobe, enhancing your tracing efficiency.
- **Interprocess eBPF Maps**: Implement userspace eBPF maps in shared userspace memory for summary aggregation or control plane communication.
- **High Compatibility**: Utilize existing eBPF toolchains like clang and libbpf to develop userspace eBPF without any modifications. Fully compatible with kernel eBPF implementations, supporting CO-RE via BTF, and offering userspace host function access.
- **Advanced Tooling**: Benefit from a cross-platform eBPF interpreter and a high-speed JIT compiler powered by LLVM. It also includes a handcrafted x86 JIT in C for limited resources.
- **No instrumentation**: Can inject eBPF runtime into any running process without the need for a restart or manual recompilation. It can run not only in Linux but also in all Unix systems, Windows, and even IoT devices.

## Quick Start

With `bpftime`, you can build eBPF applications using familiar tools like clang and libbpf, and execute them in userspace. For instance, the `malloc` eBPF program traces malloc calls using uprobe and aggregates the counts using a hash map.

To get started:

```console
make install # Install the runtime
make -C example/malloc # Build the eBPF program example
export PATH=$PATH:~/.bpftime
bpftime load ./example/malloc/malloc
```

In another shell, Run the target program with eBPF inside:

```console
$ bpftime start ./example/malloc/test
Hello malloc!
malloc called from pid 250215
continue malloc...
malloc called from pid 250215
```

You can run it again with another process and you can see the output:

```console
12:44:35 
        pid=247299      malloc calls: 10
        pid=247322      malloc calls: 10
```

Run the target program with eBPF

Alternatively, you can also run the program directly in the kernel eBPF:

```console
$ sudo example/malloc/malloc
15:38:05
        pid=30415       malloc calls: 1079
        pid=30393       malloc calls: 203
        pid=29882       malloc calls: 1076
        pid=34809       malloc calls: 8
```

## In-Depth

### Comparing with Kernel eBPF Runtime

- `bpftime` allows you to use `clang` and `libbpf` to build eBPF programs, and run them directly in this runtime. We have tested it with a libbpf version in [third_party/libbpf](third_party/libbpf).
- Some kernel helpers and kfuncs may not be available in userspace.
- It does not support direct access to kernel data structures or functions like `task_struct`.

Refer to [documents/available-features.md](documents/available-features.md) for more details.

### **How it Works**

Left: kernel eBPF | Right: userspace bpftime

![How it works](documents/bpftime.png)

The inline hook implementation is based on frida.

see [documents/how-it-works.md](documents/how-it-works.md) for details.

### **Examples & Use Cases**

We can use the bpftime userspace runtime:

Attach uprobe, uretprobe or all syscall tracepoints(currently x86 only) eBPF programs to a process or a group of processes:

- `malloc`: count the malloc calls in libc by pid
- `opensnoop`: trace file open or close syscalls in a process
- `bash_readline`: trace readline calls in bash [TODO: fix it]
- `sslsniff`: trace SSL/TLS raw text in openssl [TODO: fix it]

Examples can be found in [example](example) dir. More examples are coming soon.

> Some examples may not working now, we are fixing it. You can refer to [benchmark](benchmark) dir for more working examples.

### **Performance Benchmarks**

How is the performance of `userspace uprobe` compared to `kernel uprobes`? Let's take a look at the following benchmark results:

| Probe/Tracepoint Types | Kernel (ns)  | Userspace (ns) |
|------------------------|-------------:|---------------:|
| Uprobe                 | 4751.462610 | 445.169770    |
| Uretprobe              | 5899.706820 | 472.972220    |
| Syscall Tracepoint     | 1489.04251  | 1499.47708    |

It can be attached to functions in running process just like the kernel uprobe does.

How is the performance of LLVM JIT/AOT compared to other eBPF userspace runtimes, native code or wasm runtimes? Let's take a look at the following benchmark results:

![LLVM jit benchmark](https://github.com/eunomia-bpf/bpf-benchmark/raw/main/example-output/merged_execution_times.png?raw=true)

Across all tests, the LLVM JIT for bpftime consistently showcased superior performance. Both demonstrated high efficiency in integer computations (as seen in log2_int), complex mathematical operations (as observed in prime), and memory operations (evident in memcpy and strcmp). While they lead in performance across the board, each runtime exhibits unique strengths and weaknesses. These insights can be invaluable for users when choosing the most appropriate runtime for their specific use-cases.

see [github.com/eunomia-bpf/bpf-benchmark](https://github.com/eunomia-bpf/bpf-benchmark) for how we evaluate and details.

Hash map or ring buffer compared to kernel(TODO)

See [benchmark](benchmark) dir for detail performance benchmarks.

## Build and test

see [documents/build-and-test.md](documents/build-and-test.md) for details.

## Roadmap

`bpftime` is continuously evolving with more features in the pipeline:

- [ ] An AOT compiler for eBPF can be easily added based on the LLVM IR.
- [ ] perf event and ring buffer output support.
- [ ] more examples and usecases.
- [ ] More map types and distribution maps support.
- [ ] More program types support.

Stay tuned for more developments from this promising project! You can find `bpftime` on [GitHub](https://github.com/eunomia-bpf/bpftime).

## License

This project is licensed under the MIT License.

If you have any questions or suggestions, you can also contact yunwei356@gmail.com or wechat `yunwei2567` for more details!
