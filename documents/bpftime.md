# Bpftime: A High-Performance User-Space eBPF Runtime

Yusheng Zheng, Tong Yu

Today, we are thrilled to introduce Bpftime, a Open-sourced full-featured, high-performance eBPF runtime, designed to operate in user space. It supports eBPF kernel features like uprobe, syscall trace, can be attached to other process, having a shared interprocess map, with LLVM jit, a handcrafted x86 jit. It also can be used with existing eBPF toolchains like libbpf and clang without any changed.

It can speed up eBPF programs by 10x compared to kernel uprobes, and can be used in embedded systems, IoT, edge computing, smart contracts, and cloud-native solutions.

The github repo is: <https://github.com/eunomia-bpf/bpftime>

## Bpftime vs. Alternatives

### WebAssembly (Wasm) in User Space

While Wasm has its advantages, it also comes with limitations:

- High performance costs due to security concerns with external APIs like Wasi, which require additional validation and runtime checks, often necessitating extra memory copies.
- Manual integration needed, with embedding in compile times instead of dynamic uprobe/kprobe.
- Less adaptable to API version changes due to lack of BTF CO-RE support.
- Reliance on underlying libraries for complex operations.

### eBPF in Kernel Space

The kernel space eBPF also presents its own set of challenges:

- Kernel UProbe implementation necessitates two kernel context switches, resulting in significant performance overhead.
- Limited features and unsuitability for plugin or other use cases.
- Running eBPF programs in kernel mode requires root access, increasing the attack surface and posing risks like container escape.
- Inherent vulnerabilities in eBPF can lead to Kernel Exploits.

### Other User-Space eBPF Runtimes

There are other user-space eBPF runtimes available, such as Ubpf and Rbpf:

- Ubpf: Ubpf offers ELF parsing, a simple hash map, and JIT for arm64 and x86.
- Rbpf: Rbpf provides a helper mechanism, x86 JIT, and a VM.

However, it has several limitations:

- Complex integration and usage.
- Inability to use kernel eBPF libraries and toolchains like libbpf, bpftrace, or clang.
- Lack of attach support.
- Absence of interprocess maps.
- Limited functionality in user space.
- JIT support only for arm64 or x86.

Despite these limitations, existing user-space eBPF runtimes have been used in several innovative projects, including:

- **Qemu+uBPF**: This project combines Qemu, an open-source machine emulator and virtualizer, with uBPF to enhance its capabilities. You can check out a demonstration in this [video](https://www.youtube.com).
- **Oko**: Oko extends Open vSwitch-DPDK with BPF, enhancing tools for better integration. More details are available on its [GitHub](https://github.com/oko) page.
- **Solana**: Solana uses user-space eBPF for high-performance smart contracts. You can explore more on its [GitHub](https://github.com/solana-labs/solana) page.
- **DPDK eBPF**: DPDK eBPF provides libraries for fast packet processing, further enhanced by user-space eBPF.
- **eBPF for Windows**: This project brings eBPF toolchains and runtime to the Windows kernel, expanding the reach of eBPF.

Additionally, user-space eBPF runtimes have been discussed in academic papers like "Rapidpatch: Firmware Hotpatching for Real-Time Embedded Devices" and "Femto-Containers: Lightweight Virtualization and Fault Isolation For Small Software Functions on Low-Power IoT Microcontrollers".

These projects demonstrate the versatility and potential of user-space eBPF runtimes in diverse areas such as network plugins, edge runtime, smart contracts, hotpatching, and even Windows support. The future of eBPF is indeed promising!

## Why Bpftime?

Bpftime addresses these limitations and offers a host of powerful features:

- Runs eBPF in user space just like in the kernel, achieving a 10x speedup vs. kernel uprobes.
- Uses shared eBPF maps for data & control.
- Compatible with clang, libbpf, and existing eBPF toolchains; supports CO-RE & BTF.
- Supports `external functions`(ffi) and pointers like kfunc.
- Includes a cross-platform interpreter, fast LLVM JIT compiler, and a handcrafted x86 JIT in C for limited resources.
- Can inject eBPF runtime into any running process without the need for a restart or manual recompilation.
- Runs not only in Linux but also in all Unix systems, Windows, and even IoT devices.

## benchmark

How is the performance of `userspace uprobe` compared to `kernel uprobes`? Let's take a look at the following benchmark results:

TODO: results

It can be attached to functions in running process just like the kernel uprobe does.

How is the performance of LLVM JIT/AOT compared to other eBPF userspace runtimes, native code or wasm runtimes? Let's take a look at the following benchmark results:

You can find detail benchmark results in [https://github.com/eunomia-bpf/bpf-benchmark](https://github.com/eunomia-bpf/bpf-benchmark)

## AI for eBPF Code Generation

Bpftime is also exploring the use of AI for eBPF code generation. GPT4, with the help of AI agents, can generate eBPF code with up to 80% accuracy. More information about this can be found on:

- [NL2eBPF online website](https://gpt-2-bpftrace.vercel.app/)
- [GPTtrace](https://github.com/eunomia-bpf/GPTtrace).

## The Future of Bpftime

The Bpftime project is continuously evolving, with more features in the pipeline:

- An AOT compiler for eBPF can be easily added based on the LLVM IR.
- More map types and distribution maps support.
- User-space eBPF to speed up fuse.
- eBPF for GPU sharing programs.
- RDMA with distribution eBPF runtimes.
- User-space eBPF syscall bypass.

Bpftime is an open-source project and can be found on [GitHub](https://github.com/eunomia-bpf/bpftime).

In conclusion, Bpftime is set to revolutionize the tech sphere with its high performance, compatibility with existing eBPF toolchains, and potential for AI-enhanced code generation. Stay tuned for more developments from this promising project!
