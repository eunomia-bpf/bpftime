# bpftime uprobe performance benchmark results

*Generated 2026-09-04*

## Environment

- **x86_64 host:** Linux 6.8.0-138-generic, GCC 13.3.0
- **riscv64 host:** Linux 6.18.3+, 32-core, native hardware
- **riscv64 cross-build:** GCC 13.3.0, qemu-riscv64 user-mode
- **Build:** Release (-O2/-O3)

## 1. Kernel uprobe vs bpftime frida backend (x86_64)

Median of 3 runs, 10k iterations each.

| Probe type | Kernel (ns) | Userspace/Frida (ns) | Speedup |
|---|---:|---:|---:|
| Uprobe | 5129 | 314 | **16.3x** |
| Uretprobe | 6266 | 409 | **15.3x** |
| Uprobe+Uretprobe | 9451 | 523 | **18.1x** |

### Raw data (3 runs)

| Run | Kernel uprobe | Kernel uretprobe | Kernel both | User uprobe | User uretprobe | User both |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 5109 | 6280 | 9517 | 315 | 411 | 525 |
| 2 | 5145 | 6248 | 9361 | 313 | 406 | 521 |
| 3 | 5134 | 6269 | 9475 | 313 | 409 | 521 |

The frida-based userspace backend delivers **~16x** speedup for pure
uprobe/uretprobe hits on x86_64.

## 2. Kernel uprobe (x86_64, native)

5 runs, 100k iterations each.

| Run | Uprobe (ns) | Uretprobe (ns) | Uprobe+Uretprobe (ns) |
|---|---:|---:|---:|
| 1 | 5150 | 5980 | 6450 |
| 2 | 5133 | 5979 | 6456 |
| 3 | 5328 | 5984 | 7156 |
| 4 | 5112 | 5959 | 6430 |
| 5 | 5137 | 5969 | 6494 |
| **Avg** | **5172** | **5974** | **6597** |

Baseline (no probe): ~5 ns.

## 3. Trap backend (riscv64, native hardware)

5 runs, 100k iterations each, on a 32-core riscv64 board.

| Run | Uprobe (ns) | Uretprobe (ns) | Uprobe+Uretprobe (ns) |
|---|---:|---:|---:|
| 1 | 5574 | 8412 | 8446 |
| 2 | 5611 | 8449 | 8530 |
| 3 | 5603 | 8429 | 8451 |
| 4 | 5631 | 8485 | 8545 |
| 5 | 5598 | 8439 | 8507 |
| **Avg** | **5603** | **8443** | **8496** |

Baseline (no probe): ~3.5 ns.

## 4. Cross-architecture comparison

Note: different CPUs and architectures, comparison is for order-of-magnitude
reference only. A same-machine riscv64 kernel uprobe vs trap comparison is
pending (requires libelf-dev on the target).

| Probe type | x86_64 kernel uprobe (ns) | riscv64 trap backend (ns) | Ratio |
|---|---:|---:|---:|
| Uprobe | 5172 | 5603 | 1.08x |
| Uretprobe | 5974 | 8443 | 1.41x |
| Uprobe+Uretprobe | 6597 | 8496 | 1.29x |

The trap backend on riscv64 native hardware is in the same order of magnitude
as x86_64 kernel uprobe. For uprobe-only, the overhead difference is ~8%.
Uretprobe is more expensive due to double signal delivery (SIGTRAP on entry
and return).

## 5. Trap backend (riscv64, qemu-user)

Mean of 5 runs, 50k iterations each. qemu-user emulates signals with
significant overhead — these numbers are NOT representative of real hardware.

| Probe type | qemu-user (ns) | Overhead vs baseline (ns) |
|---|---:|---:|
| Baseline (no probe) | 65 | -- |
| Uprobe | 6185 | 6119 |
| Uretprobe | 9522 | 9457 |
| Uprobe+Uretprobe | 9993 | 9928 |

### Multi-thread (4 threads, uprobe+uretprobe, qemu-user)

| Thread | Avg (ns) |
|---|---:|
| 1 | 13990 |
| 2 | 14014 |
| 3 | 14023 |
| 4 | 13905 |

## 6. otel-ebpf-profiler

Deferred — requires Go 1.25+ not available in current environment.
