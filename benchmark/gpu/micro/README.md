# GPU Micro-Benchmark Framework

A flexible benchmarking framework for measuring GPU eBPF instrumentation overhead.

## Quick Start

**Important:** Run from the bpftime root directory (not from `benchmark/gpu/micro/`)

### Run Built-in Micro-benchmarks
```bash
cd /path/to/bpftime
python3 benchmark/gpu/micro/run_cuda_bench.py benchmark/gpu/micro/bench_config.json
```
Tests various eBPF probe types: empty, entry, exit, both, ringbuf, timer, etc.

### Run Example Programs
```bash
cd /path/to/bpftime
python3 benchmark/gpu/micro/run_cuda_bench.py benchmark/gpu/micro/bench_config_examples.json
```
Tests real-world eBPF examples: cuda-counter, mem_trace, threadhist, etc.

## Configuration Files

### `bench_config.json`
Built-in micro-benchmarks testing:
- **Baseline**: No eBPF (native CUDA performance)
- **Empty probe**: Minimal eBPF overhead
- **Entry/Exit probes**: Kernel entry/exit instrumentation
- **GPU Ringbuf**: GPU-side event logging
- **Global timer**: GPU timer measurements
- **Per-GPU-thread array**: Per-thread data structures
- **Memtrace**: Memory access tracing
- **CPU map operations**: Array/Hash map operations from GPU

### `bench_config_examples.json`
Real-world examples from `example/gpu/`:
- **cuda-counter**: Count kernel invocations
- **mem_trace**: Trace memory access patterns
- **threadhist**: Thread execution histogram
- **launchlate**: Kernel launch latency
- **kernelretsnoop**: Kernel return value snooping

## Workload Presets

| Workload | Elements | Iterations | Threads | Blocks |
|----------|----------|------------|---------|--------|
| tiny     | 32       | 10000      | 32      | 1      |
| small    | 1000     | 10000      | 256     | 4      |
| medium   | 10000    | 10000      | 256     | 40     |
| large    | 100000   | 1000       | 512     | 196    |
| xlarge   | 1000000  | 1000       | 512     | 1954   |

## Output Files

After running benchmarks, outputs are saved to `benchmark/gpu/micro/`:

**For micro-benchmarks (bench_config.json):**
- **`micro_result.md`**: Markdown-formatted results
- **`micro_result.json`**: Raw JSON data
- **`micro_bench.log`**: Detailed execution log

**For examples (bench_config_examples.json):**
- **`micro_example_result.md`**: Markdown-formatted results
- **`micro_example_result.json`**: Raw JSON data
- **`micro_example_bench.log`**: Detailed execution log

## Configuration Structure

Both config files use the same structure:

```json
{
  "workload_presets": {
    "minimal": "32 3 32 1"
  },
  "test_cases": [
    {
      "name": "Test Name",
      "probe_binary_cmd": "path/to/probe [args]",
      "workload": "minimal",
      "baseline": "Baseline (minimal)"
    }
  ]
}
```

- **Empty `probe_binary_cmd`**: Runs baseline (no eBPF)
- **With probe path**: Runs with eBPF instrumentation
- **Paths are relative to bpftime root**: `benchmark/gpu/micro/cuda_probe entry` or `example/gpu/mem_trace/mem_trace`

**Important:** All tests use `benchmark/gpu/micro/vec_add` for consistent benchmarking.

## Architecture

```
┌─────────────────────────────────────────────┐
│         run_cuda_bench.py                   │
│  - Load config                              │
│  - Run baselines (no eBPF)                  │
│  - Run eBPF tests                           │
│  - Collect metrics                          │
│  - Generate reports                         │
└─────────────────────────────────────────────┘
              │
              ├──► bench_config.json
              │    (built-in micro-benchmarks)
              │
              ├──► bench_config_examples.json
              │    (example programs)
              │
              └──► Custom configs
```

## Test Execution Flow

1. **Baseline**: Run vec_add directly (no eBPF)
2. **eBPF Tests**:
   - Start eBPF probe in background (with syscall-server)
   - Run vec_add with agent preloaded
   - Collect timing data
   - Terminate probe
3. **Calculate Overhead**: Compare against matching baseline
4. **Generate Reports**: Output to MD/JSON/LOG

## Dependencies

- Python 3.x
- CUDA toolkit
- bpftime built with GPU support
- eBPF examples compiled (if running examples config)

## Troubleshooting

**"Custom probe not found"**: Ensure examples are built
```bash
cd ../../example/gpu
make
```

**"No vec_add_args specified"**: Each test case needs a workload preset

**High overhead numbers**: Check that baseline is running correctly without eBPF

## Contributing

To add new test cases:
1. Add to appropriate config file
2. Reference existing baseline
3. Specify workload preset
4. For custom programs, add to `example_programs` section

## License

Same as bpftime project
