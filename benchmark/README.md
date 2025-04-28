# Benchmark

This directory contains benchmarks and experiments for the bpftime project, including:

- Scripts for running experiments and generating figures
- Benchmark environments for different use cases
- Test code for performance evaluation

## Getting Started

### Install Dependencies

Please refer to our manual in [bpftime build and test documentation](https://eunomia.dev/bpftime/documents/build-and-test/) for installing dependencies or using the Docker image.

### Basic Usage

Check out the [bpftime usage documentation](https://eunomia.dev/bpftime/documents/usage/) for basic usage instructions.

### Run All Experiments

```sh
cd /path/to/bpftime
make benchmark
./benchmark/run_all_experiment.sh
```

## Experiments Overview

### Experiment 1: Micro-benchmarks

#### Part 1: bpftime vs eBPF

Performance comparison including:

- Uprobe/uretprobe (see `./uprobe/`)
- Memory read/write operations (see `./uprobe/`)
- Map operations (see `./uprobe/`)
- Syscall tracepoint (see `./syscall/`)
- MPK enable/disable (see `./mpk/`)

#### Part 2: Execution Engine Efficiency

See the code used in our [bpf-benchmark repository](https://github.com/eunomia-bpf/bpf-benchmark).

#### Part 3: Load Latency

The measurement tool is located in [../tools/cli/main.cpp](../tools/cli/main.cpp).

### Experiment 2: SSL/TLS Traffic Inspection (sslsniff)

- Environment and results: See [./ssl-nginx/](./ssl-nginx/)
- Example code: See [../example/sslsniff](../example/sslsniff)

### Experiment 3: System Call Counting (syscount)

- Environment and results: See [./syscount-nginx/](./syscount-nginx/)
- Example code: See [../example/syscount](../example/syscount)

### Experiment 4: Nginx Plugin/Module

- Implementation code: See [../example/attach_implementation](../example/attach_implementation)
- Benchmark scripts are included in the implementation directory

### Experiment 5: DeepFlow

Performance evaluation for DeepFlow integration - see [./deepflow/](./deepflow/) directory.

### Experiment 6: FUSE (Filesystem in Userspace)

FUSE-related benchmarks - see [./fuse/](./fuse/) directory.

### Experiment 7: Redis Durability Tuning

Redis durability tuning benchmarks - see [./redis-durability-tuning/](./redis-durability-tuning/) directory.

### Experiment 8: Compatibility

Various compatibility examples - see [../example](../example) directory.
