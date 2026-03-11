# Multi-GPU Load Balance Monitor

**Zero-modification black-box monitoring** of multi-GPU CUDA workloads using
bpftime's eBPF GPU attach.

## Why This Example Matters

Traditional GPU profiling tools (Nsight, nvprof) require:
- Explicit setup for each target program
- Exclusive profiling mode (interferes with production workloads)
- Fixed, predefined metrics — no custom analysis logic
- Per-GPU profiling sessions that must be manually correlated

**bpftime's eBPF GPU attach** provides capabilities these tools cannot:

| Capability | Nsight/nvprof | bpftime eBPF |
|---|---|---|
| Source modification required | No | **No** |
| Black-box binary monitoring | Limited | **Yes — `bpftime load ./any_cuda_binary`** |
| Custom programmable probes | No | **Yes — arbitrary eBPF logic on GPU** |
| Cross-GPU unified maps | No | **Yes — all GPUs share maps via UVA** |
| Per-GPU discrimination from GPU side | Manual setup | **Automatic via gridDim.x** |
| Production-safe always-on monitoring | No (heavy overhead) | **Yes (lightweight probes)** |
| Dynamic attach/detach | No | **Yes** |

The key insight: bpftime injects eBPF probes into GPU kernel PTX at load time.
All GPUs write to the **same shared eBPF maps** through CUDA Unified Virtual
Addressing (UVA), so a single probe automatically provides a unified cross-GPU
view — something that requires significant manual effort with traditional tools.

## How It Works

The CUDA workload (`multi_gpu_vec_add`) is a **black-box target** — it is
monitored without any source code modification. The eBPF probe
(`multi_gpu_probe`) is a separate program that bpftime automatically injects
into all GPU kernels at runtime.

```
┌─────────────────────────────────────────────────────┐
│  Target: multi_gpu_vec_add (unmodified black box)   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ GPU 0    │ │ GPU 1    │ │ GPU 2    │ ...         │
│  │ N elems  │ │ 2N elems │ │ 3N elems │ (imbalanced)│
│  └──────────┘ └──────────┘ └──────────┘             │
│        │             │             │                 │
│        ▼             ▼             ▼                 │
│  ┌──────────────────────────────────────────────┐   │
│  │ Host-side: CUDA event timing (standard)      │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  bpftime eBPF probe (auto-injected into kernel PTX) │
│  ┌──────────────────────────────────────────────┐   │
│  │ kprobe/kretprobe on vectorAdd                │   │
│  │ • Per-block globaltimer measurement          │   │
│  │ • Per-GPU stats via gridDim.x identification │   │
│  │ • Cross-GPU latency histogram (7 buckets)    │   │
│  │ • All GPUs → same shared maps (UVA)          │   │
│  └──────────────────────────────────────────────┘   │
│        │                                            │
│        ▼                                            │
│  ┌──────────────────────────────────────────────┐   │
│  │ GPU-Internal Monitor: per-GPU + aggregate    │   │
│  │ block latency — invisible to CUDA events     │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Per-GPU Identification from Inside the GPU

The eBPF probe uses `gridDim.x` (PTX register `%nctaid.x`, helper ID 508) to
identify which GPU each block belongs to. Since the imbalanced workload assigns
different numbers of blocks to each GPU, `gridDim.x` serves as a natural
per-GPU identifier — purely from GPU-internal data, without any host
coordination.

## Files

| File | Description |
|------|-------------|
| `multi_gpu_vec_add.cu` | CUDA workload with intentional imbalance + CUDA event timing |
| `multi_gpu_probe.bpf.c` | eBPF probe: per-GPU + aggregate block timing, histogram |
| `multi_gpu_probe.c` | Userspace loader with live GPU-internal dashboard |
| `Makefile` | Build rules |

## Build & Run

```bash
# Build CUDA workload
nvcc -cudart shared -o multi_gpu_vec_add multi_gpu_vec_add.cu

# Run standalone (host-side load balance dashboard only)
./multi_gpu_vec_add 4 10     # 4 GPUs, 10 iterations

# Build eBPF probe
make

# Run with bpftime (host-side + GPU-internal monitoring)
# Note: works on ANY CUDA binary — no source modification needed
export PATH=$PATH:~/.bpftime/
bpftime load ./multi_gpu_vec_add 4 20
bpftime start ./multi_gpu_probe
```

## Example Output

**Host-side** (`multi_gpu_vec_add` — standard CUDA event timing):
```
╔═══════════════════════════════════════════════════════════════╗
║  MULTI-GPU LOAD BALANCE DASHBOARD  -  Iteration 5           ║
╠═══════════════════════════════════════════════════════════════╣
║  GPU │  Elements  │  Time (ms)  │  Relative  │  Bar         ║
╠═══════════════════════════════════════════════════════════════╣
║   0  │    524288  │      0.142  │    25.3%   │  ##........  ║
║   1  │   1048576  │      0.283  │    50.4%   │  #####.....  ║
║   2  │   1572864  │      0.421  │    74.9%   │  #######...  ║
║   3  │   2097152  │      0.562  │   100.0%   │  ########## ║
╠═══════════════════════════════════════════════════════════════╣
║  Min:   0.142 ms   Max:   0.562 ms   Avg:   0.352 ms       ║
║  Stdev: 0.154 ms   Imbalance:  74.7%   Utilization:  62.6% ║
╚═══════════════════════════════════════════════════════════════╝
```

**GPU-internal** (`multi_gpu_probe` via eBPF — **only possible with bpftime**):
```
╔══════════════════════════════════════════════════════════════════╗
║  GPU-INTERNAL BLOCK LATENCY MONITOR (eBPF)     14:23:05        ║
╠══════════════════════════════════════════════════════════════════╣
║  Kernel invocations: 40          Blocks profiled: 20480        ║
╠══════════════════════════════════════════════════════════════════╣
║  Block Duration:  min=1203    ns  avg=8421      ns  max=45621 ns║
╠══════════════════════════════════════════════════════════════════╣
║  Per-GPU Block Timing (by grid size):                          ║
║  Grid  2048 │ avg     8421 ns │   2048 blks │ ##........  25%  ║
║  Grid  4096 │ avg     8502 ns │   4096 blks │ #####.....  50%  ║
║  Grid  6144 │ avg     8388 ns │   6144 blks │ #######...  75%  ║
║  Grid  8192 │ avg     8450 ns │   8192 blks │ ########## 100%  ║
╠══════════════════════════════════════════════════════════════════╣
║  Latency Histogram (per-block distribution):                    ║
║        <1us │######                        │ 2041              ║
║      1-10us │##############################│ 15234             ║
║    10-100us │########                      │ 3102              ║
║   100us-1ms │#                             │ 103               ║
║      1-10ms │                              │ 0                 ║
║    10-100ms │                              │ 0                 ║
║      100ms+ │                              │ 0                 ║
╚══════════════════════════════════════════════════════════════════╝

  These metrics are measured INSIDE the GPU via eBPF globaltimer.
  Host-side CUDA events cannot observe per-block latency distribution.
```

The **Per-GPU Block Timing** section shows each GPU's average block execution
time, identified by grid size (`gridDim.x`). This data comes entirely from
GPU-internal eBPF measurements — no host-side timing involved. All GPUs
contribute to the same shared eBPF maps via UVA, making cross-GPU comparison
automatic.
