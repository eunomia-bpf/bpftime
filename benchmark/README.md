# Benchmark and performance evaluation for bpftime

The [benchmark](https://github.com/eunomia-bpf/bpftime/tree/master/benchmark) directory contains benchmarks and experiments for the bpftime project, including:

- Scripts for running experiments and generating figures
- Benchmark environments for different use cases
- Test code for performance evaluation

The benchmark is also tested for each commit in the CI: [https://github.com/eunomia-bpf/bpftime/tree/master/.github/workflows/benchmarks.yml](https://github.com/eunomia-bpf/bpftime/tree/master/.github/workflows/benchmarks.yml)

The result will be published in [https://eunomia-bpf.github.io/bpftime/benchmark/uprobe/results.html](https://eunomia-bpf.github.io/bpftime/benchmark/uprobe/results.html)

You can check our OSDI paper [Extending Applications Safely and Efficiently](https://www.usenix.org/conference/osdi25/presentation/zheng-yusheng) for more benchmark detail.

## Getting Started

### Install Dependencies

Please refer to our manual in [bpftime build and test documentation](https://eunomia.dev/bpftime/documents/build-and-test/)
for installing dependencies or using the Docker image.

The benchmark experiment scripts may automatically install dependencies
and clone repos from the GitHub. Make sure you have Network access.

Run the experiments needs you have a Linux kernel with eBPF support, at lease 4 cores,
and 16GB memory on x86_64 architecture.

### Basic Usage

Check out the [bpftime usage documentation](https://eunomia.dev/bpftime/documents/usage/)
for basic usage instructions. For the detail usage, please refer to each experiment directory.

### Run All Experiments

before running the experiments, you also need to install some additional dependencies for the python scripts:

```sh
cd /path/to/bpftime
pip install -r benchmark/requirements.txt
```

then you can build and run the experiments by:

```sh
make benchmark # build the benchmark
make run-all-benchmark # run all benchmarks
```

(build time: 10min - 20min)

See the makefile for the details of the commands.

You can also check the CI for how we build the experiments and run them in [.github/workflows/build-benchmarks.yml](https://github.com/eunomia-bpf/bpftime/tree/master/.github/workflows/benchmarks.yml).

## Experiments Overview

### Experiment 1: Micro-benchmarks

This experiment measures the performance overhead and latency differences between bpftime and traditional kernel eBPF across various operations and use cases.

An example would be like:

> *Generated on 2025-04-30 03:01:13*. Environment

- **OS:** Linux 6.11.0-24-generic
- **CPU:** Intel(R) Core(TM) Ultra 7 258V (4 cores, 4 threads)
- **Memory:** 15.07 GB
- **Python:** 3.12.7

Core Uprobe Performance Summary

| Operation | Kernel Uprobe | Userspace Uprobe | Speedup |
|-----------|---------------|------------------|---------|
| __bench_uprobe | 2561.57 | 190.02 | 13.48x |
| __bench_uretprobe | 3019.45 | 187.10 | 16.14x |
| __bench_uprobe_uretprobe | 3119.28 | 191.63 | 16.28x |

Kernel vs Userspace eBPF Detailed Comparison

| Operation | Environment | Min (ns) | Max (ns) | Avg (ns) | Std Dev |
|-----------|-------------|----------|----------|----------|---------|
| __bench_array_map_delete | Kernel | 2725.99 | 3935.98 | 3237.62 | 359.11 |
| __bench_array_map_delete | Userspace | 2909.07 | 3285.52 | 3096.46 | 114.99 |
| __bench_array_map_lookup | Kernel | 2641.18 | 4155.25 | 2992.88 | 402.00 |
| __bench_array_map_lookup | Userspace | 3354.17 | 3724.05 | 3486.81 | 108.63 |
| __bench_array_map_update | Kernel | 9945.97 | 14917.03 | 12225.93 | 1508.60 |
| __bench_array_map_update | Userspace | 4398.82 | 4841.92 | 4629.57 | 152.57 |
| __bench_hash_map_delete | Kernel | 18560.92 | 27069.99 | 22082.68 | 2295.90 |
| __bench_hash_map_delete | Userspace | 9557.35 | 11240.72 | 10253.54 | 473.67 |
| __bench_hash_map_lookup | Kernel | 10181.58 | 13742.86 | 12375.69 | 1142.61 |
| __bench_hash_map_lookup | Userspace | 20580.46 | 23586.77 | 22152.81 | 969.63 |
| __bench_hash_map_update | Kernel | 43969.13 | 61331.16 | 53376.22 | 5497.51 |
| __bench_hash_map_update | Userspace | 21172.05 | 25878.44 | 23992.67 | 1264.81 |
| __bench_per_cpu_array_map_delete | Kernel | 2782.47 | 3716.44 | 3183.09 | 287.23 |
| __bench_per_cpu_array_map_delete | Userspace | 2865.53 | 3409.70 | 3114.67 | 140.65 |
| __bench_per_cpu_array_map_lookup | Kernel | 2773.47 | 4176.10 | 3170.42 | 416.42 |
| __bench_per_cpu_array_map_lookup | Userspace | 6269.58 | 7395.49 | 7018.47 | 345.91 |
| __bench_per_cpu_array_map_update | Kernel | 10662.37 | 15923.08 | 12326.39 | 1522.21 |
| __bench_per_cpu_array_map_update | Userspace | 15592.15 | 17505.63 | 16528.99 | 553.50 |
| __bench_per_cpu_hash_map_delete | Kernel | 19709.29 | 26844.96 | 21994.95 | 2243.80 |
| __bench_per_cpu_hash_map_delete | Userspace | 55954.89 | 76124.07 | 65603.07 | 5986.58 |
| __bench_per_cpu_hash_map_lookup | Kernel | 10783.48 | 15208.46 | 12315.21 | 1525.86 |
| __bench_per_cpu_hash_map_lookup | Userspace | 48033.46 | 57481.09 | 50651.83 | 2503.34 |
| __bench_per_cpu_hash_map_update | Kernel | 31072.46 | 43163.81 | 35580.60 | 3748.51 |
| __bench_per_cpu_hash_map_update | Userspace | 73661.69 | 79157.12 | 76526.24 | 1868.13 |
| __bench_read | Kernel | 22506.85 | 30934.20 | 25865.43 | 3018.32 |
| __bench_read | Userspace | 1491.75 | 1862.13 | 1653.45 | 101.66 |
| __bench_uprobe | Kernel | 2130.54 | 4389.26 | 2561.57 | 628.77 |
| __bench_uprobe | Userspace | 166.54 | 232.13 | 190.02 | 16.11 |
| __bench_uprobe_uretprobe | Kernel | 2658.28 | 3859.19 | 3119.28 | 311.45 |
| __bench_uprobe_uretprobe | Userspace | 179.61 | 202.69 | 191.63 | 9.64 |
| __bench_uretprobe | Kernel | 2581.48 | 3916.19 | 3019.45 | 359.75 |
| __bench_uretprobe | Userspace | 175.54 | 196.49 | 187.10 | 7.66 |
| __bench_write | Kernel | 22783.52 | 31415.49 | 26478.92 | 2787.90 |
| __bench_write | Userspace | 1406.01 | 1802.50 | 1542.49 | 106.23 |

#### Part 1: bpftime vs eBPF

Performance comparison including:

- Uprobe/uretprobe (see [./uprobe/](https://github.com/eunomia-bpf/bpftime/tree/master/benchmark/uprobe/))
- Memory read/write operations (see [./uprobe/](https://github.com/eunomia-bpf/bpftime/tree/master/benchmark/uprobe/))
- Map operations (see [./uprobe/](https://github.com/eunomia-bpf/bpftime/tree/master/benchmark/uprobe/))
- Embedded VM in your program without hooking (see [./uprobe/](https://github.com/eunomia-bpf/bpftime/tree/master/benchmark/uprobe/) and the code in [test_embed.c](https://github.com/eunomia-bpf/bpftime/tree/master/benchmark/test_embed.c))
- Syscall tracepoint (see [./syscall/](https://github.com/eunomia-bpf/bpftime/tree/master/benchmark/syscall/))
- MPK enable/disable (see [./mpk/](https://github.com/eunomia-bpf/bpftime/tree/master/benchmark/mpk/))

You can check each directory for the details of the experiments, how to run them and the results.

(20min - 30 min computation time)

#### Part 2: Execution Engine Efficiency

This part evaluates the execution performance of different eBPF virtual machines and JIT compilers to compare their efficiency in running eBPF programs.

See the code used in our [bpf-benchmark repository](https://github.com/eunomia-bpf/bpf-benchmark).

#### Part 3: Load Latency

This part measures the time required to load and attach eBPF programs.

The measurement tool is located in [../tools/cli/main.cpp](https://github.com/eunomia-bpf/bpftime/tree/master/tools/cli/main.cpp).

### Experiment 2: SSL/TLS Traffic Inspection (sslsniff)

This experiment demonstrates bpftime's capability to intercept and inspect SSL/TLS traffic in real-time by hooking into OpenSSL functions within nginx, measuring both performance impact and functionality.

- Environment and results: See [./ssl-nginx/](https://github.com/eunomia-bpf/bpftime/tree/master/benchmark/ssl-nginx/)
- Example code: See [../example/tracing/sslsniff](https://github.com/eunomia-bpf/bpftime/tree/master/example/tracing/sslsniff)

### Experiment 3: System Call Counting (syscount)

This experiment evaluates bpftime's ability to trace and count system calls made by applications like nginx, comparing the overhead and accuracy with kernel-based tracing.

- Environment and results: See [./syscount-nginx/](https://github.com/eunomia-bpf/bpftime/tree/master/benchmark/syscount-nginx/)
- Example code: See [../example/tracing/syscount](https://github.com/eunomia-bpf/bpftime/tree/master/example/tracing/syscount)

### Experiment 4: Nginx Plugin/Module

This experiment showcases how bpftime can be integrated as a plugin or module within nginx.

- Implementation code: See [../example/attach_implementation](https://github.com/eunomia-bpf/bpftime/tree/master/example/attach_implementation)
- Benchmark scripts are included in the implementation directory

### Experiment 5: DeepFlow

This experiment measures the performance impact of integrating bpftime with DeepFlow, an observability platform, to evaluate how userspace eBPF affects network monitoring and tracing workloads.

Performance evaluation for DeepFlow integration - see [./deepflow/](https://github.com/eunomia-bpf/bpftime/tree/master/benchmark/deepflow/) directory.

### Experiment 6: FUSE (Filesystem in Userspace)

This experiment evaluates bpftime's performance when instrumenting FUSE-based filesystems to cache syscall results.

FUSE-related benchmarks - see [./fuse/](https://github.com/eunomia-bpf/bpftime/tree/master/benchmark/fuse/) directory.

### Experiment 7: Redis Durability Tuning

This experiment demonstrates how bpftime can be used to dynamically tune Redis durability settings at runtime, measuring the performance benefits of userspace extensions for database optimization.

Redis durability tuning benchmarks - see [./redis-durability-tuning/](https://github.com/eunomia-bpf/bpftime/tree/master/benchmark/redis-durability-tuning/) directory.

### Experiment 8: Compatibility

This experiment validates that existing eBPF programs can run seamlessly on both kernel eBPF and bpftime without modification, demonstrating the compatibility and portability of the userspace eBPF runtime.

Various compatibility examples that can run on both kernel eBPF and bpftime - see [../example](https://github.com/eunomia-bpf/bpftime/tree/master/example) directory.
