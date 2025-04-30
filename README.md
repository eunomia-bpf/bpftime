# eGPU: Extending eBPF Programmability and Observability to GPUs

[![Build and Test VM](https://github.com/eunomia-bpf/bpftime-super/actions/workflows/test-vm.yml/badge.svg)](https://github.com/eunomia-bpf/bpftime-super/actions/workflows/test-vm.yml)
[![Build and test runtime](https://github.com/eunomia-bpf/bpftime-super/actions/workflows/test-runtime.yml/badge.svg)](https://github.com/eunomia-bpf/bpftime-super/actions/workflows/test-runtime.yml)
[![DOI](https://img.shields.io/badge/arXiv-2312.05531-blue.svg)](https://asplos.dev/pdf/bpftime_super.pdf)

`bpftime-super` is the first system to dynamically offload eBPF instrumentation and bytecode directly onto running GPU kernels using real-time PTX injection, significantly reducing instrumentation overhead compared to existing methods.

## Installation

```bash
git clone https://github.com/eunomia-bpf/eGPU.git
cd bpftime-super
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
| **Dynamic PTX injection**               | At load‑time we JIT eBPF → PTX and patch it into the resident kernel | < 1 µs probe overhead on micro‑benchmarks        |
| **Shared eBPF maps across CPU & GPU**   | `boost::managed_shared_memory` exposes the same map to host threads *and* device code | Zero‑copy metrics exchange                       |
| **User‑space verifier & JIT (bpftime)** | All safety checks stay in user space; no root privileges required | Fast iteration & lower attack surface            |
| **Hot‑swap instrumentation**            | Add / remove probes while kernels keep running               | Debug live services without downtime             |
| **CXL.mem latency modelling**           | Optional delay injection emulates tier‑2 memory              | Prototype far‑memory systems on today’s hardware |

------

### Project highlights

- **Low overhead:** < 5 % runtime impact on memory‑bound kernels up to 128 KB access size (see Fig. 2 of the paper). 
- **Open ecosystem:** Works with standard eBPF tooling—`clang`, `bpftool`, `bpftrace`.
- **Future‑proof:** Design anticipates Grace‑Hopper architectures & CXL memory pools.

```txt
@article{yang2025bpftimesuper,
      title={eGPU: Extending eBPF Programmability and Observability to GPUs}, 
      author={Yiwei Yang, Yu Tong, Yusheng Zheng, Andrew Quinn},
      year={2025},
      archivePrefix={4th Workshop on Heterogeneous Composable and Disaggregated Systems},
      primaryClass={cs.OS}
}
```
