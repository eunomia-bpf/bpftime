# bpftime: Userspace eBPF runtime for Observability, Network & General extensions Framework

[![Build and Test VM](https://github.com/eunomia-bpf/bpftime/actions/workflows/test-vm.yml/badge.svg)](https://github.com/eunomia-bpf/bpftime/actions/workflows/test-vm.yml)
[![Build and test runtime](https://github.com/eunomia-bpf/bpftime/actions/workflows/test-runtime.yml/badge.svg)](https://github.com/eunomia-bpf/bpftime/actions/workflows/test-runtime.yml)
[![DOI](https://zenodo.org/badge/676866666.svg)](https://doi.org/10.48550/arXiv.2311.07923)

`bpftime` is a High-Performance userspace eBPF runtime and General Extension Framework designed for userspace. It enables faster Uprobe, USDT, Syscall hooks, XDP, and more event sources by bypassing the kernel and utilizing an optimized compiler like `LLVM`.

📦 [Key Features](#key-features) \
🔨 [Quick Start](#quick-start) \
🔌 [Examples & Use Cases](#examples--use-cases) \
⌨️ [Linux Plumbers 23 talk](https://lpc.events/event/17/contributions/1639/) \
📖 [Slides](https://eunomia.dev/bpftime/documents/userspace-ebpf-bpftime-lpc.pdf) \
📚 [Arxiv preprint](https://arxiv.org/abs/2311.07923)

[**Checkout our documents in eunomia.dev!**](https://eunomia.dev/bpftime/)

bpftime is not `userspace eBPF VM`, it's a userspace runtime framework includes everything to run eBPF in userspace: `loader`, `verifier`, `helpers`, `maps`, `ufunc` and multiple `events` such as Observability, Network, Policy or Access Control. It has multiple VM backend options support. For eBPF VM only, please see [llvmbpf](https://github.com/eunomia-bpf/llvmbpf).

> ⚠️ **Note**: `bpftime` is currently under active development and refactoring towards v2. It may contain bugs or unstable API. Please use it with caution. For more details, check our [roadmap](#roadmap). We'd love to hear your feedback and suggestions! Feel free to open an issue or [Contact us](#contact-and-citations).

## Why bpftime? What's the design Goal?

- **Performance Gains**: Achieve better performance by `bypassing the kernel` (e.g., via `Userspace DBI` or `Network Drivers`), with more configurable, optimized and more arch supported JIT/AOT options like `LLVM`, while maintaining compatibility with Linux kernel eBPF.
- **Cross-Platform Compatibility**: Enables `eBPF functionality and large ecosystem` where kernel eBPF is unavailable, such as on older or alternative operating systems, or where kernel-level permissions are restricted, without changing your tool.
- **Flexible and General Extension Language & Runtime for Innovation**: eBPF is designed for innovation, evolving into a General Extension Language & Runtime in production that supports very diverse use cases. `bpftime`'s modular design allows easy integration as a library for adding new events and program types without touching kernel. Wishing it could enable rapid prototyping and exploration of new features!

## Key Features

- **Dynamic Binary rewriting**: Run eBPF programs in userspace, attaching them to `Uprobes` and `Syscall tracepoints`: **No manual instrumentation or restart required!**. It can `trace` or `change` the execution of a function, `hook` or `filter` all syscalls of a process safely, and efficiently with an eBPF userspace runtime. Can inject eBPF runtime into any running process without the need for a restart or manual recompilation.
- **Performance**: Experience up to a `10x` speedup in Uprobe overhead compared to kernel uprobe and uretprobe. Read/Write userspace memory is also faster than kernel eBPF.
- **Interprocess eBPF Maps**: Implement userspace `eBPF maps` in shared userspace memory for summary aggregation or control plane communication.
- **Compatibility**: use `existing eBPF toolchains` like clang, libbpf and bpftrace to develop userspace eBPF application without any modifications. Supporting CO-RE via BTF, and offering userspace `ufunc` access.
- **Multi JIT Support**: Support [llvmbpf](https://github.com/eunomia-bpf/llvmbpf), a high-speed `JIT/AOT` compiler powered by LLVM, or using `ubpf JIT` and INTERPRETER. The vm can be built as `a standalone library` like ubpf.
- **Run with kernel eBPF**: Can load userspace eBPF from kernel, and using kernel eBPF maps to cooperate with kernel eBPF programs like kprobes and network filters.
- **Integrate with AF_XDP or DPDK**: Run your `XDP` network applications with better performance in userspace just like in kernel!(experimental)

## Components

- [`vm`](https://github.com/eunomia-bpf/bpftime/tree/master/vm): The eBPF VM and JIT compiler for bpftime, you can choose from [bpftime LLVM JIT/AOT compiler](https://github.com/eunomia-bpf/llvmbpf) and [ubpf](https://github.com/iovisor/ubpf). The [llvm-based vm](https://github.com/eunomia-bpf/llvmbpf) in bpftime can also be built as a standalone library and integrated into other projects, similar to ubpf.
- [`runtime`](https://github.com/eunomia-bpf/bpftime/tree/master/runtime): The userspace runtime for eBPF, including the maps, helpers, ufuncs and other runtime safety features.
- [`Attach events`](https://github.com/eunomia-bpf/bpftime/tree/master/attach): support attaching eBPF programs to `Uprobes`, `Syscall tracepoints`, `XDP` and other events with bpf_link, and also the driver event sources.
- [`verifier`](https://github.com/eunomia-bpf/bpftime/tree/master/bpftime-verifier): Support using [PREVAIL](https://github.com/vbpf/ebpf-verifier) as userspace verifier, or using `Linux kernel verifier` for better results.
- [`Loader`](https://github.com/eunomia-bpf/bpftime/tree/master/runtime/syscall-server): Includes a `LD_PRELOAD` loader library in userspace can work with current eBPF toolchain and library without involving any kernel, Another option is [daemon](https://github.com/eunomia-bpf/bpftime/tree/master/daemon) when Linux eBPF is available.

## Quick Start: Uprobe

With `bpftime`, you can build eBPF applications using familiar tools like clang and libbpf, and execute them in userspace. For instance, the [`malloc`](https://github.com/eunomia-bpf/bpftime/tree/master/example/malloc) eBPF program traces malloc calls using uprobe and aggregates the counts using a hash map.

You can refer to [eunomia.dev/bpftime/documents/build-and-test](https://eunomia.dev/bpftime/documents/build-and-test) for how to build the project, or using the container images from [GitHub packages](https://github.com/eunomia-bpf/bpftime/pkgs/container/bpftime).

To get started, you can build and run a libbpf based eBPF program starts with `bpftime` cli:

```console
make -C example/malloc # Build the eBPF program example
export PATH=$PATH:~/.bpftime/
bpftime load ./example/malloc/malloc
```

In another shell, Run the target program with eBPF inside:

```console
$ bpftime start ./example/malloc/victim
Hello malloc!
malloc called from pid 250215
continue malloc...
malloc called from pid 250215
```

You can also dynamically attach the eBPF program with a running process:

```console
$ ./example/malloc/victim & echo $! # The pid is 101771
[1] 101771
101771
continue malloc...
continue malloc...
```

And attach to it:

```console
$ sudo bpftime attach 101771 # You may need to run make install in root
Inject: "/root/.bpftime/libbpftime-agent.so"
Successfully injected. ID: 1
```

You can see the output from original program:

```console
$ bpftime load ./example/malloc/malloc
...
12:44:35 
        pid=247299      malloc calls: 10
        pid=247322      malloc calls: 10
```

Alternatively, you can also run our sample eBPF program directly in the kernel eBPF, to see the similar output. This can be an example of how bpftime can work compatibly with kernel eBPF.

```console
$ sudo example/malloc/malloc
15:38:05
        pid=30415       malloc calls: 1079
        pid=30393       malloc calls: 203
        pid=29882       malloc calls: 1076
        pid=34809       malloc calls: 8
```

See [eunomia.dev/bpftime/documents/usage](https://eunomia.dev/bpftime/documents/usage) for more details.

## Examples & Use Cases

For more examples and details, please refer to [eunomia.dev/bpftime/documents/examples/](https://eunomia.dev/bpftime/documents/examples/) webpage.

Examples including:

- [Minimal examples](https://github.com/eunomia-bpf/bpftime/tree/master/example/minimal) of eBPF programs.
- eBPF `Uprobe/USDT` tracing and `syscall tracing`:
  - [sslsniff](https://github.com/eunomia-bpf/bpftime/tree/master/example/sslsniff) for trace SSL/TLS unencrypted data.
  - [opensnoop](https://github.com/eunomia-bpf/bpftime/tree/master/example/opensnoop) for trace file open syscalls.
  - More [bcc/libbpf-tools](https://github.com/eunomia-bpf/bpftime/tree/master/example/libbpf-tools).
  - Run with [bpftrace](https://github.com/eunomia-bpf/bpftime/tree/master/example/bpftrace) commands or scripts.
- [error injection](https://github.com/eunomia-bpf/bpftime/tree/master/example/error-inject): change function behavior with `bpf_override_return`.
- Use the eBPF LLVM JIT/AOT vm as [a standalone library](https://github.com/eunomia-bpf/llvmbpf/tree/main/example).
- Userspace [XDP with DPDK and AF_XDP](https://github.com/userspace-xdp/userspace-xdp)

## In-Depth

### **How it Works**

bpftime supports two modes:

#### Running in userspace only

Left: original kernel eBPF | Right: bpftime

![How it works](https://eunomia.dev/bpftime/documents/bpftime.png)

In this mode, bpftime can run eBPF programs in userspace without kernel, so it can be ported into low version of Linux or event other systems, and running without root permissions. It relies on a [userspace verifier](https://github.com/vbpf/ebpf-verifier) to ensure the safety of eBPF programs.

#### Run with kernel eBPF

![documents/bpftime-kernel.png](https://eunomia.dev/bpftime/documents/bpftime-kernel.png)

In this mode, bpftime can run together with kernel eBPF. It can load eBPF programs from kernel, and using kernel eBPF maps to cooperate with kernel eBPF programs like kprobes and network filters.

#### Instrumentation implementation

Current hook implementation is based on binary rewriting and the underly technique is inspired by:

- Userspace function hook: [frida-gum](https://github.com/frida/frida-gum)
- Syscall hooks: [zpoline](https://www.usenix.org/conference/atc23/presentation/yasukata) and [pmem/syscall_intercept](https://github.com/pmem/syscall_intercept).

The hook can be easily replaced with other DBI methods or frameworks, or add more hook mechanisms in the future.

See our draft arxiv paper [bpftime: userspace eBPF Runtime for Uprobe, Syscall and Kernel-User Interactions](https://arxiv.org/abs/2311.07923) for details.

### **Performance Benchmarks**

How is the performance of `userspace uprobe` compared to `kernel uprobes`?

| Probe/Tracepoint Types | Kernel (ns)  | Userspace (ns) |
|------------------------|-------------:|---------------:|
| Uprobe                 | 3224.172760  | 314.569110     |
| Uretprobe              | 3996.799580  | 381.270270     |
| Syscall Tracepoint     | 151.82801    | 232.57691      |
| Manually Instrument    | Not avaliable |  110.008430   |

It can be attached to functions in running process just like the kernel uprobe does.

How is the performance of LLVM JIT/AOT compared to other eBPF userspace runtimes, native code or wasm runtimes?

![LLVM jit benchmark](https://github.com/eunomia-bpf/bpf-benchmark/raw/main/example-output/jit_execution_times.png?raw=true)

Across all tests, the LLVM JIT for bpftime consistently showcased superior performance. Both demonstrated high efficiency in integer computations (as seen in log2_int), complex mathematical operations (as observed in prime), and memory operations (evident in memcpy and strcmp). While they lead in performance across the board, each runtime exhibits unique strengths and weaknesses. These insights can be invaluable for users when choosing the most appropriate runtime for their specific use-cases.

see [github.com/eunomia-bpf/bpf-benchmark](https://github.com/eunomia-bpf/bpf-benchmark) for how we evaluate and details.

Hash map or ring buffer compared to kernel(TODO)

See [benchmark](https://github.com/eunomia-bpf/bpftime/tree/master/benchmark) dir for detail performance benchmarks.

### Comparing with Kernel eBPF Runtime

- `bpftime` allows you to use `clang` and `libbpf` to build eBPF programs, and run them directly in this runtime, just like normal kernel eBPF. We have tested it with a libbpf version in [third_party/libbpf](https://github.com/eunomia-bpf/bpftime/tree/master/third_party/libbpf). No specify libbpf or clang version needed.
- Some kernel helpers and kfuncs may not be available in userspace.
- It does not support direct access to kernel data structures or functions like `task_struct`.

Refer to [eunomia.dev/bpftime/documents/available-features](https://eunomia.dev/bpftime/documents/available-features) for more details.

## Build and test

See [eunomia.dev/bpftime/documents/build-and-test](https://eunomia.dev/bpftime/documents/build-and-test) for details.

## Roadmap

`bpftime` is continuously evolving with more features in the pipeline:

- [ ] Keep compatibility with the evolving kernel
- [ ] Refactor for General Extension Framework
- [ ] Trying to refactor, bug fixing for `Production`.
- [ ] More examples and usecases:
  - [X] Userspace Network Driver on userspace eBPF
  - [X] Hotpatch userspace application
  - [X] Error injection and filter syscall
  - [X] Syscall bypassing, batching
  - [X] Userspace Storage Driver on userspace eBPF
  - [ ] etc...

Stay tuned for more developments from this promising project! You can find `bpftime` on [GitHub](https://github.com/eunomia-bpf/bpftime).

## License

This project is licensed under the MIT License.

## Contact and citations

Have any questions or suggestions on future development? Free free to open an issue or contact
<yunwei356@gmail.com> !

Our arxiv preprint: <https://arxiv.org/abs/2311.07923>

```txt
@misc{zheng2023bpftime,
      title={bpftime: userspace eBPF Runtime for Uprobe, Syscall and Kernel-User Interactions}, 
      author={Yusheng Zheng and Tong Yu and Yiwei Yang and Yanpeng Hu and XiaoZheng Lai and Andrew Quinn},
      year={2023},
      eprint={2311.07923},
      archivePrefix={arXiv},
      primaryClass={cs.OS}
}
```

## Acknowledgement

eunomia-bpf community is sponsored by [PLCT Lab](https://plctlab.github.io/) from [ISCAS](http://english.is.cas.cn/au/).

Thanks for other sponsors and discussions help building this project: [Prof. Marios Kogias](https://marioskogias.github.io/) from Imperial College London, [Prof. Xiaozheng lai](https://www2.scut.edu.cn/cs/2017/0129/c22285a327654/page.htm) from SCUT, [Prof lijun chen](http://www.xiyou.edu.cn/info/2394/67845.htm) from XUPT,
[Prof. Qi Li](https://sites.google.com/site/qili2012/) from THU [NISL Lab](https://netsec.ccert.edu.cn/en/), and Linux eBPF maintainers in the LPC 23 eBPF track.
