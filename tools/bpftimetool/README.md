# bpftimetool

Command-line tool for inspecting and managing bpftime shared memory state.

## Overview

`bpftimetool` provides utilities to:
- **Export** shared memory state to JSON for inspection or backup
- **Import** eBPF objects from JSON into shared memory
- **Run** eBPF programs with performance benchmarking
- **Remove** shared memory segments system-wide

## Installation

After building bpftime, the tool is available at:
```bash
~/.bpftime/bpftimetool
# Or add to PATH
export PATH=$PATH:~/.bpftime/
```

## Command Reference

### export - Dump Shared Memory to JSON

Export all eBPF objects (programs, maps, links) from shared memory to a JSON file:

```bash
bpftimetool export <filename>
```

**Example:**
```console
$ bpftimetool export state.json
[info] Global shm constructed. shm_open_type 1 for bpftime_maps_shm
[info] bpf_map_handler name=.rodata.str1.1 found at 3
[info] find prog fd=4 name=do_uprobe_trace
[info] bpf_perf_event_handler found at 5
```

The JSON file contains:
- eBPF program bytecode and metadata
- Map definitions and attributes
- Perf event attachments
- Handler file descriptors

**Use cases:**
- Debugging runtime state
- Backup before modifications
- Analysis and reverse engineering
- Sharing test cases

### import - Load JSON into Shared Memory

Import eBPF objects from a JSON file into bpftime shared memory:

```bash
bpftimetool import <filename>
```

**Example:**
```console
$ SPDLOG_LEVEL=Debug bpftimetool import minimal.json
[info] Global shm constructed. shm_open_type 3 for bpftime_maps_shm
[info] import handler fd 3 {"attr":{...},"name":".rodata.str1.1","type":"bpf_map_handler"}
[info] import handler type bpf_prog_handler fd 4
[info] import handler type bpf_perf_event_handler fd 5
```

**Use cases:**
- Restore previous state
- Load pre-configured environments
- Testing and CI/CD pipelines
- Cross-system deployment

### run - Execute and Benchmark Programs

Run an eBPF program from shared memory with performance measurement:

```bash
bpftimetool run <id> <data_file> [repeat N] [type RUN_TYPE]

Arguments:
  id          Program ID in shared memory
  data_file   Input data file (program context)

Options:
  repeat N           Run N times and report average (default: 1)
  type RUN_TYPE      Execution mode: JIT | AOT | INTERPRET (default: JIT)
```

**Examples:**
```bash
# Run program once with JIT
bpftimetool run 4 input.bin

# Benchmark with 10000 iterations
bpftimetool run 4 input.bin repeat 10000

# Compare execution modes
bpftimetool run 4 input.bin repeat 1000 type JIT
bpftimetool run 4 input.bin repeat 1000 type AOT
bpftimetool run 4 input.bin repeat 1000 type INTERPRET
```

**Output:**
```console
Running eBPF program with id 4 and data in file input.bin
Repeat N: 10000 with run type JIT
Time taken: 1250 ns
Return value: 0
```

**Execution modes:**
- `JIT`: LLVM JIT compilation (fast compilation, good performance)
- `AOT`: Pre-compiled native code (best performance, requires prior compilation)
- `INTERPRET`: Bytecode interpreter (slowest, no compilation overhead)

### remove - Clean Up Shared Memory

Remove bpftime shared memory segments system-wide:

```bash
bpftimetool remove
```

**Warning:** This destroys all loaded eBPF programs, maps, and state for all processes using bpftime.

**Use cases:**
- Clean up after crashes
- Reset test environment
- Free system resources

## Common Workflows

### Debugging Workflow

1. **Capture current state:**
```bash
bpftimetool export debug_state.json
```

2. **Inspect JSON to find program IDs and understand state:**
```bash
cat debug_state.json | jq '.handlers[] | select(.type=="bpf_prog_handler")'
```

3. **Test specific program:**
```bash
bpftimetool run 4 test_input.bin
```

### Performance Analysis Workflow

Compare execution modes for a program:

```bash
# Create test data
echo -n "test data" > input.bin

# Benchmark different modes
echo "=== Interpreter ==="
bpftimetool run 4 input.bin repeat 10000 type INTERPRET

echo "=== JIT ==="
bpftimetool run 4 input.bin repeat 10000 type JIT

echo "=== AOT (after compilation) ==="
bpftime-aot compile  # Compile first
bpftimetool run 4 input.bin repeat 10000 type AOT
```

### Backup and Restore

```bash
# Backup before making changes
bpftimetool export backup_$(date +%Y%m%d_%H%M%S).json

# ... make changes ...

# Restore if needed
bpftimetool remove
bpftimetool import backup_20250930_120000.json
```

### Cross-System Testing

Export on development machine, import on test machine:

```bash
# On dev machine
bpftimetool export production_config.json
scp production_config.json test-server:~/

# On test machine
bpftimetool import production_config.json
```

## JSON Format

The exported JSON contains an array of handlers with this structure:

```json
{
  "handlers": [
    {
      "fd": 3,
      "type": "bpf_map_handler",
      "name": ".rodata.str1.1",
      "attr": {
        "map_type": 2,
        "key_size": 4,
        "value_size": 21,
        "max_entries": 1,
        "flags": 128
      }
    },
    {
      "fd": 4,
      "type": "bpf_prog_handler",
      "name": "do_uprobe_trace",
      "attr": {
        "type": 0,
        "cnt": 16,
        "insns": "<hex-encoded-bytecode>",
        "attach_fds": [5]
      }
    },
    {
      "fd": 5,
      "type": "bpf_perf_event_handler",
      "attr": {
        "type": 6,
        "pid": -1,
        "_module_name": "example/minimal/victim",
        "offset": 4457
      }
    }
  ]
}
```

### Handler Types

- **bpf_map_handler**: eBPF maps (hash, array, ringbuf, etc.)
- **bpf_prog_handler**: eBPF programs with bytecode
- **bpf_perf_event_handler**: Uprobe/kprobe/tracepoint attachments
- **bpf_link_handler**: BPF links connecting programs to hooks

## Environment Variables

- `SPDLOG_LEVEL`: Set logging level (trace, debug, info, warn, error, critical)
  ```bash
  SPDLOG_LEVEL=debug bpftimetool import state.json
  ```

## Troubleshooting

**Q: "Global shm not found" error**
A: No bpftime processes are running. Start a process with bpftime first:
```bash
LD_PRELOAD=~/.bpftime/libbpftime-syscall-server.so your_program
```

**Q: "Invalid id not exist" when running**
A: Use `export` to see available program IDs. The ID may have been destroyed or never created.

**Q: "AOT instructions not found" error**
A: The program hasn't been AOT-compiled yet. Run `bpftime-aot compile` first.

**Q: JSON import fails**
A: Ensure the JSON format matches the expected schema. Use `export` to see correct format.

## Performance Notes

- **Export**: Fast, reads shared memory without copying large data
- **Import**: Creates all objects in shared memory, may take time for large programs
- **Run**: Performance depends on execution mode:
  - INTERPRET: ~100-500ns per iteration overhead
  - JIT: ~10-50ns per iteration overhead (first run includes compilation)
  - AOT: ~5-20ns per iteration overhead (pre-compiled)

## Integration with bpftime

`bpftimetool` works with the bpftime runtime's shared memory:

```bash
# Terminal 1: Start target with bpftime
LD_PRELOAD=~/.bpftime/libbpftime-syscall-server.so ./my_app

# Terminal 2: Inspect state
bpftimetool export current_state.json

# Terminal 3: Run benchmarks
bpftimetool run 4 test_data.bin repeat 100000 type JIT
```

## See Also

- [bpftime-aot](https://github.com/eunomia-bpf/bpftime/tree/master/tools/aot) - AOT compiler for eBPF programs
- [bpftime documentation](https://github.com/eunomia-bpf/bpftime) - Main project documentation
- [Shared memory architecture](https://github.com/eunomia-bpf/bpftime/blob/master/runtime/include/bpftime_shm.hpp) - Implementation details
