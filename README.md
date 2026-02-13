# bpftime: Userspace eBPF runtime for Observability, Network, GPU & General extensions Framework

[![Build and Test VM](https://github.com/eunomia-bpf/bpftime/actions/workflows/test-vm.yml/badge.svg)](https://github.com/eunomia-bpf/bpftime/actions/workflows/test-vm.yml)
[![Build and test runtime](https://github.com/eunomia-bpf/bpftime/actions/workflows/test-runtime.yml/badge.svg)](https://github.com/eunomia-bpf/bpftime/actions/workflows/test-runtime.yml)
[![DOI](https://zenodo.org/badge/676866666.svg)](https://doi.org/10.48550/arXiv.2311.07923)

`bpftime` is a High-Performance userspace eBPF runtime and General Extension Framework designed for userspace. It enables faster Uprobe, USDT, Syscall hooks, XDP, and more event sources by bypassing the kernel and utilizing an optimized compiler like `LLVM`.

📦 [Key Features](#key-features) \
🔨 [Quick Start](#quick-start) \
🔌 [Examples & Use Cases](#examples--use-cases) \
⌨️ [Linux Plumbers 23 talk](https://lpc.events/event/17/contributions/1639/) \
📖 [Slides](https://eunomia.dev/bpftime/documents/userspace-ebpf-bpftime-lpc.pdf) \
📚 [OSDI '25 Paper](https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng)

[**Checkout our documents in eunomia.dev**](https://eunomia.dev/bpftime/) and [Deepwiki](https://deepwiki.com/eunomia-bpf/bpftime)!

bpftime is not `userspace eBPF VM`, it's a userspace runtime framework includes everything to run eBPF in userspace: `loader`, `verifier`, `helpers`, `maps`, `ufunc` and multiple `events` such as Observability, Network, Policy or Access Control. It has multiple VM backend options support. For eBPF VM only, please see [llvmbpf](https://github.com/eunomia-bpf/llvmbpf).

> ⚠️ **Note**: `bpftime` is currently under active development and refactoring towards v2. It may contain bugs or unstable API. Please use it with caution. For more details, check our [roadmap](#roadmap). We'd love to hear your feedback and suggestions! Feel free to open an issue or [Contact us](#contact-and-citations).

## Why bpftime? What's the design Goal?

- **Performance Gains**: Achieve better performance by `bypassing the kernel` (e.g., via `Userspace DBI` or `Network Drivers`), with more configurable, optimized and more arch supported JIT/AOT options like `LLVM`, while maintaining compatibility with Linux kernel eBPF.
- **Cross-Platform Compatibility**: Enables `eBPF functionality and large ecosystem` where kernel eBPF is unavailable, such as on older or alternative operating systems, or where kernel-level permissions are restricted, without changing your tool.
- **Flexible and General Extension Language & Runtime for Innovation**: eBPF is designed for innovation, evolving into a General Extension Language & Runtime in production that supports very diverse use cases. `bpftime`'s modular design allows easy integration as a library for adding new events and program types without touching kernel. Wishing it could enable rapid prototyping and exploration of new features!

## Key Features

- **Dynamic Binary rewriting**: Run eBPF programs in userspace, attaching them to `Uprobes`, `Syscall tracepoints` and inside `GPU` kernel: **No manual instrumentation or restart required!**. It can `trace` or `change` the execution of a function, `hook` or `filter` all syscalls of a process safely, and efficiently with an eBPF userspace runtime. Can inject eBPF runtime into any running process without the need for a restart or manual recompilation.
- **Performance**: Experience up to a `10x` speedup in Uprobe overhead compared to kernel uprobe and uretprobe， up to a 10x faster than `NVbit`. Read/Write userspace memory is also faster than kernel eBPF.
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

You can refer to [eunomia.dev/bpftime/documents/build-and-test](https://eunomia.dev/bpftime/documents/build-and-test) or the `installation.md` in the repo for how to build the project. Or You can using the container images from [GitHub packages](https://github.com/eunomia-bpf/bpftime/pkgs/container/bpftime).

To get started, you can build and run a libbpf based eBPF program starts with `bpftime` cli:

```console
make -C example/malloc # Build the eBPF program example
export PATH=$PATH:~/.bpftime/
bpftime load ./example/malloc/malloc
```

In another shell, Run the target program with eBPF inside:

```console
$ bpftime start ./example/malloc/victim
malloc called from pid 250215
continue malloc...
malloc called from pid 250215
continue malloc...
```

You should always run the `load` first then run the `start` command, or the eBPF program will not be attached.

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

For more examples and details, please refer to [eunomia.dev/bpftime/documents/examples/](https://eunomia.dev/bpftime/documents/examples/) webpage and [example](https://github.com/eunomia-bpf/bpftime/tree/master/example/) dir.

## In-Depth

### **How it Works**

bpftime supports two modes:

#### Running in userspace only

Left: original kernel eBPF | Right: bpftime

![How it works](https://eunomia.dev/bpftime/documents/bpftime.png)

In this mode, bpftime can run eBPF programs in userspace without kernel, so it can be ported into low version of Linux or even other systems, and running without root permissions. It relies on a [userspace verifier](https://github.com/vbpf/ebpf-verifier) to ensure the safety of eBPF programs.

#### Run with kernel eBPF

![documents/bpftime-kernel.png](https://eunomia.dev/bpftime/documents/bpftime-kernel.png)

In this mode, bpftime can run together with kernel eBPF. It can load eBPF programs from kernel, and using kernel eBPF maps to cooperate with kernel eBPF programs like kprobes and network filters.

#### Instrumentation implementation

Current hook implementation is based on binary rewriting and the underly technique is inspired by:

- Userspace function hook: [frida-gum](https://github.com/frida/frida-gum)
- Syscall hooks: [zpoline](https://www.usenix.org/conference/atc23/presentation/yasukata) and [pmem/syscall_intercept](https://github.com/pmem/syscall_intercept).
- GPU hooks: our new implementation by converting eBPF into PTX and injecting into GPU kernels. See [attach/nv_attach_impl](https://github.com/eunomia-bpf/bpftime/tree/master/attach/nv_attach_impl) for more details.
- XDP with DPDK. See the [uXDP paper](https://dl.acm.org/doi/10.1145/3748355.3748360) for more details.

The hook can be easily replaced with other DBI methods or frameworks, to make it a general extension framework. See our OSDI '25 paper [Extending Applications Safely and Efficiently](https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng) for details.

### **Performance Benchmarks**

see [github.com/eunomia-bpf/bpf-benchmark](https://github.com/eunomia-bpf/bpf-benchmark) for how we evaluate and details.

### Comparing with Kernel eBPF Runtime

- `bpftime` allows you to use `clang` and `libbpf` to build eBPF programs, and run them directly in this runtime, just like normal kernel eBPF. We have tested it with a libbpf version in [third_party/libbpf](https://github.com/eunomia-bpf/bpftime/tree/master/third_party/libbpf). No specify libbpf or clang version needed.
- Some kernel helpers and kfuncs may not be available in userspace.
- It does not support direct access to kernel data structures or functions like `task_struct`.

Refer to [eunomia.dev/bpftime/documents/available-features](https://eunomia.dev/bpftime/documents/available-features) for more details.

## Build and test

See [eunomia.dev/bpftime/documents/build-and-test](https://eunomia.dev/bpftime/documents/build-and-test) for details.

## License

This project is licensed under the MIT License.

## Contact and citations

Have any questions or suggestions on future development? Feel free to open an issue or contact
<yunwei356@gmail.com> !

Our OSDI '25 paper: <https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng>

```txt
@inproceedings{zheng2025extending,
  title={Extending Applications Safely and Efficiently},
  author={Zheng, Yusheng and Yu, Tong and Yang, Yiwei and Hu, Yanpeng and Lai, Xiaozheng and Williams, Dan and Quinn, Andi},
  booktitle={19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25)},
  pages={557--574},
  year={2025}
}
```

## Acknowledgement

eunomia-bpf community is sponsored by [PLCT Lab](https://plctlab.github.io/) from [ISCAS](http://english.is.cas.cn/au/).

Thanks for other sponsors and discussions help building this project: [Prof. Marios Kogias](https://marioskogias.github.io/) from Imperial College London, [Prof. Xiaozheng lai](https://www2.scut.edu.cn/cs/2017/0129/c22285a327654/page.htm) from SCUT, [Prof lijun chen](http://www.xiyou.edu.cn/info/2394/67845.htm) from XUPT,
[Prof. Qi Li](https://sites.google.com/site/qili2012/) from THU [NISL Lab](https://netsec.ccert.edu.cn/en/), and Linux eBPF maintainers in the LPC 23 eBPF track.
