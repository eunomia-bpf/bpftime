# Memory Protection Keys (MPK) Benchmark

This benchmark compares the performance of BPFtime with and without Memory Protection Keys (MPK) enabled to assess the performance impact of this security feature.

## What is MPK?

Memory Protection Keys (MPK) is a CPU feature available in recent Intel processors that allows user-space processes to protect regions of memory from access by other parts of the same process. In the context of BPFtime, MPK is used to isolate eBPF programs from the host application, providing an additional layer of security.

## How to Run the Benchmark

### Prerequisites

- Linux system with Intel CPU supporting MPK (generally Intel Core 7th generation or newer)
- BPFtime built both with and without MPK support

### Build Instructions

The benchmark requires two builds of BPFtime:

1. A standard build (without MPK) in `build/`
2. An MPK-enabled build in `build-mpk/`

You can use the `build.sh` script in this directory to create both builds automatically.

### Running the Benchmark

```bash
cd path/to/bpftime
python benchmark/mpk/benchmark.py
```

The script will:

1. Start a server process for each configuration (MPK and standard)
2. Run multiple benchmark tests with both configurations
3. Generate a markdown report of the results in `benchmark/mpk/results.md`
4. Save detailed benchmark data in JSON format in `benchmark/mpk/benchmark-output.json`

## Example Results

An example of benchmark results can be found in [example_results.md](example_results.md).

