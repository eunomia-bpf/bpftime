# eGPU: Extending eBPF Programmability and Observability to GPUs

[![Build and Test VM](https://github.com/eunomia-bpf/eGPU/actions/workflows/build-benchmarks.yml/badge.svg)](https://github.com/eunomia-bpf/eGPU/actions/workflows/build-benchmarks.yml)
[![Build and test runtime](https://github.com/eunomia-bpf/eGPU/actions/workflows/test-attach.yml/badge.svg)](https://github.com/eunomia-bpf/eGPU/actions/workflows/test-attach.yml)
[![DOI](https://img.shields.io/badge/DOI-10.1145/3723851.3726984-1f57b6?style=flat&link=https://dl.acm.org/doi/pdf/10.1145/3723851.3726984)](https://dl.acm.org/doi/pdf/10.1145/3723851.3726984)

`eGPU` is the first system to dynamically offload eBPF instrumentation and bytecode directly onto running GPU kernels using real-time PTX injection, significantly reducing instrumentation overhead compared to existing methods.

## Installation

```bash
git clone https://github.com/eunomia-bpf/eGPU.git
cd eGPU
docker run -dit --gpus all \
                -v.:/root \
                --privileged --network=host --ipc=host \
                --name egpu yangyw12345/egpu:latest
make release
```
To support Intel GPU or AMD GPU, please use [ZLUDA](https://github.com/vickiegpt/ZLUDA) as backend.

## eGPU – Extending eBPF Programmability & Observability to GPUs

**eGPU** is the first open‑source framework that lets you run eBPF programs *inside* live GPU kernels.
 By JIT‑translating eBPF byte‑code to NVIDIA PTX at runtime, eGPU injects ultra‑lightweight probes directly into running kernels without pausing or recompiling them. The result is micro‑second‑level visibility into kernel execution, memory transfers and heterogeneous orchestration with **minimal overhead**. ​

------

### Why eGPU?

- Traditional GPU profilers (CUPTI, NVBit, …) either interrupt kernels or impose high per‑event cost.
- Linux eBPF offers elegant, safe instrumentation—but only for CPUs.
- Modern AI & HPC workloads need continuous telemetry across **both** CPU and GPU to catch memory stalls, launch gaps, and anomalous behavior in production.

eGPU bridges that gap by marrying the flexibility of eBPF with the parallel fire‑power of GPUs. 

------

### Core capabilities

| Capability                              | How it works                                                 | Benefit                                          |
| --------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| **Dynamic PTX injection**               | At load-time we JIT eBPF → PTX and patch it into the resident kernel | < 1 µs probe overhead on micro-benchmarks        |
| **Shared eBPF maps across CPU & GPU**   | `boost::managed_shared_memory` exposes the same map to host threads *and* device code | Zero-copy metrics exchange                       |
| **User-space verifier & JIT (bpftime)** | All safety checks stay in user space; no root privileges required | Fast iteration & lower attack surface            |
| **Hot-swap instrumentation**            | Add / remove probes while kernels keep running               | Debug live services without downtime             |
| **Isolation & protection domain**       | PTX-level fencing and gsafe sandbox partition instrumentation into a protected domain | Prevents instrumentation from corrupting application state and enforces security |

------

### Project highlights

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

In this mode, bpftime can run eBPF programs in userspace without kernel, so it can be ported into low version of Linux or even other systems, and running without root permissions. It relies on a [userspace verifier](https://github.com/vbpf/ebpf-verifier) to ensure the safety of eBPF programs.

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
@inproceedings{yang2025egpu,
  title={eGPU: Extending eBPF Programmability and Observability to GPUs},
  author={Yang, Yiwei and Yu, Tong and Zheng, Yusheng and Quinn, Andrew},
  booktitle={Proceedings of the 4th Workshop on Heterogeneous Composable and Disaggregated Systems},
  pages={73--79},
  year={2025}
}
```
