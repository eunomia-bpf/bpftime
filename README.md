# bpftime: Userspace unprivileged eBPF runtime for fast Uprobe & Syscall Tracing

`bpftime` is an unprivileged eBPF runtime designed to operate in userspace, offering rapid Uprobe and Syscall tracing capabilities.

## Features

- **Unprivileged Userspace eBPF**: Seamlessly run eBPF programs in userspace, attaching them to Uprobes and Syscall tracepoints just like in the kernel.
- **Performance**: Achieve up to 10x speedup compared to kernel uprobe and uretprobe.
- **Interprocess eBPF Maps**: Utilize userspace eBPF maps in shared userspace memory for summary aggregation or control plane communication.
- **Compatibility**: Use existing eBPF toolchains like clang and libbpf without any modifications. Compatible with kernel eBPF implementations without requiring privileged access.
- **Advanced Tooling**: Comes with a cross-platform eBPF interpreter and a high-speed JIT compiler powered by LLVM.

> ⚠️ **Note**: This project is actively under development. The API might undergo changes in upcoming releases.

## Quick Start

With `bpftime`, you can build eBPF applications using familiar tools like clang and libbpf, and execute them in userspace. For instance, the `malloc` eBPF program traces malloc calls using uprobe and aggregates the counts using a hash map.

To get started:

1. Navigate to the example directory:

   ```bash
   cd example/malloc
   ```

2. Build the eBPF program:

   ```bash
   make
   ```

3. Execute the libbpf tracing program:

   ```bash
   LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so ./malloc
   ```

4. Run the target program with eBPF:

   ```bash
   LD_PRELOAD=build/runtime/agent/libbpftime-agent.so ./test
   ```

Alternatively, you can run the program directly in the kernel:

```console
$ sudo example/malloc/malloc
15:38:05
        pid=30415       malloc calls: 1079
        pid=30393       malloc calls: 203
        pid=29882       malloc calls: 1076
        pid=34809       malloc calls: 8
```

## In-Depth

### **How it Works**

TODO

### **Examples & Use Cases**

TODO

### **Performance Benchmarks**

TODO

## Building from Source

### Dependencies

Install the required packages:

```bash
sudo apt install libelf1 libelf-dev zlib1g-dev make git libboost-dev cmake
git submodule update --init --recursive
```

### Compilation

Build the complete runtime:

```bash
make build
```

For a lightweight build without the runtime (only core library and LLVM JIT):

```bash
make build-core
make build-llvm
```

## Testing

Run the test suite to validate the implementation:

```bash
make test
```

## License

This project is licensed under the MIT License.
