# BPFtime Syscall Benchmark Results

Benchmark run at: **2025-04-30 04:19:36**

## System Information

- OS: Linux yunwei 6.11.0-24-generic #24-Ubuntu SMP PREEMPT_DYNAMIC Fri Mar 14 18:13:56 UTC 2025 x86_64
- Hostname: yunwei
- Number of runs per configuration: 10

## Benchmark Results

### Native (No tracing)

| Metric | Value |
| ------ | ----- |
| Average time usage (mean) | **404.44 ns** |
| Average time usage (median) | 403.88 ns |
| Standard deviation | 5.64 |
| Min | 397.46 ns |
| Max | 414.01 ns |

**Individual runs (ns):**
`403.16, 404.6, 414.01, 412.81, 398.68, 407.07, 397.46, 401.33, 405.7, 399.62`

### Kernel Tracepoint Syscall

| Metric | Value |
| ------ | ----- |
| Average time usage (mean) | **438.92 ns** |
| Average time usage (median) | 437.71 ns |
| Standard deviation | 9.04 |
| Min | 425.70 ns |
| Max | 455.27 ns |

**Individual runs (ns):**
`425.7, 442.86, 436.82, 445.81, 437.9, 437.51, 446.62, 433.26, 455.27, 427.45`

### Userspace BPF Syscall

| Metric | Value |
| ------ | ----- |
| Average time usage (mean) | **484.65 ns** |
| Average time usage (median) | 483.22 ns |
| Standard deviation | 12.17 |
| Min | 463.26 ns |
| Max | 509.91 ns |

**Individual runs (ns):**
`490.16, 475.32, 483.09, 478.91, 483.35, 463.26, 509.91, 492.71, 481.97, 487.79`

## Comparison Results

### Overhead Compared to Native

| Configuration | Overhead |
| ------------ | -------- |
| Kernel Tracepoint Syscall | **8.52%** |
| Userspace BPF Syscall | **19.83%** |

### Userspace BPF vs Kernel Tracepoint

Userspace BPF syscall has **10.42%** more overhead than kernel tracepoint

## Summary

This benchmark compares three configurations for syscall handling:

1. **Native**: No tracing or interception
2. **Kernel Tracepoint**: Traditional kernel-based syscall tracking
3. **Userspace BPF**: BPFtime's userspace syscall interception

Each configuration was run multiple times to ensure statistical significance. Lower numbers represent better performance.
