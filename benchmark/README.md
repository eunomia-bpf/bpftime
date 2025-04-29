# Benchmark

This directory contains benchmarks and experiments for the bpftime project, including:

- Scripts for running experiments and generating figures
- Benchmark environments for different use cases
- Test code for performance evaluation

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
make benchmark
./benchmark/run_all_experiment.sh
```

You can also check the CI for how we build the experiments and run some of them in [.github/workflows/build-benchmarks.yml](../.github/workflows/benchmarks.yml).

## Experiments Overview

### Experiment 1: Micro-benchmarks

#### Part 1: bpftime vs eBPF

Performance comparison including:

- Uprobe/uretprobe (see [./uprobe/](./uprobe/))
- Memory read/write operations (see [./uprobe/](./uprobe/))
- Map operations (see [./uprobe/](./uprobe/))
- Syscall tracepoint (see [./syscall/](./syscall/))
- MPK enable/disable (see [./mpk/](./mpk/))

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

Various compatibility examples that can run on both kernel eBPF and bpftime - see [../example](../example) directory.
