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
- Example code: See [../example/sslsniff](https://github.com/eunomia-bpf/bpftime/tree/master/example/sslsniff)

### Experiment 3: System Call Counting (syscount)

This experiment evaluates bpftime's ability to trace and count system calls made by applications like nginx, comparing the overhead and accuracy with kernel-based tracing.

- Environment and results: See [./syscount-nginx/](https://github.com/eunomia-bpf/bpftime/tree/master/benchmark/syscount-nginx/)
- Example code: See [../example/syscount](https://github.com/eunomia-bpf/bpftime/tree/master/example/syscount)

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
