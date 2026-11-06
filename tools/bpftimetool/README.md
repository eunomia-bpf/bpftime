# bpftimetool

Command-line tool for inspecting and managing bpftime shared memory state.

## Overview

`bpftimetool` provides utilities to:
- **Export** object definitions to JSON for inspection or reuse
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

All commands accept `--help`. Command-line syntax and numeric validation errors
produce a diagnostic and exit status 1 before accessing shared memory.
Validation of JSON contents and runtime objects remains part of command execution.
File descriptors and program IDs must be decimal integers in 0..INT_MAX;
repeat counts and CUDA counts/dimensions must be in 1..INT_MAX. A leading
`+` or whitespace before a number remains accepted; trailing characters,
including whitespace, are rejected.

The documented `run ... repeat N type RUN_TYPE` syntax remains supported.
`--repeat N`, `--repeat=N`, `--type RUN_TYPE` and `--type=RUN_TYPE` are also
accepted, and the two options may appear in either order. Repeated options
are rejected, including mixtures such as `repeat 2 --repeat 3`; repeated
`type` no longer silently overwrites the earlier value. The default remains
one JIT execution.

As with other argparse-based commands, names starting with `-` are treated
as options. Use a path such as `./-input.bin` or `./--help` for such files;
do not rely on a bare `--` separator with the bundled argparse version.

### load - Load One JSON Object

```bash
bpftimetool load <fd> <JSON>
```

Load one JSON-encoded handler at the specified descriptor. Quote the JSON
as one shell argument. This differs from `import`, which reads a JSON file.

### export - Dump Shared Memory to JSON

Export the supported eBPF object-definition graph to a JSON file:

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
- eBPF program bytecode, optional AOT bytes, and metadata
- Map definitions and attributes
- Perf event attachments
- Handler file descriptors

**Use cases:**
- Debugging runtime state
- Inspecting definitions before modifications
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
- Recreating object definitions in an empty segment
- Load pre-configured environments
- Testing and CI/CD pipelines
- Testing in a compatible environment

### Import/export contract

JSON stores object definitions and their references. It does **not** capture map
entries, perf/ring-buffer contents, epoll registrations, memfd contents, or a live
process checkpoint. Importing an array or hash map creates a new empty map. Map
descriptor aliases are independent definitions; shared map identity is not
preserved. Initialize map data separately before executing imported programs.

Both operations require quiescent shared memory: stop concurrent users from
creating, deleting, or changing handlers while importing or exporting. They do
not provide a concurrent snapshot or transaction isolation from other processes.
Keep process-local custom map callback registration unchanged during import.

`import` and `load` use the same strict decoder. The entire input is validated,
declared link and legacy attachment references are checked, and resources are
prepared before handlers are published.
Every requested descriptor must be free. An error leaves existing handlers and
their data unchanged and publishes no partial graph. A failed CLI import may
still create an empty shared-memory segment. Import does not reset the runtime
configuration of an existing segment.

`export` encodes the complete supported graph before writing a temporary regular
file in the destination directory. It replaces the destination only after the
write and close succeed. Failure leaves an existing file unchanged. New files
have mode `0600`; replacement copies only the existing owner/group/other
read, write, and execute mode bits (`0777`). Ownership, ACLs, and special mode
bits are not copied. Symlinks, multiply linked files, the active segment being
exported (including path aliases), and non-regular targets (including
`/dev/full`, devices, and directories) are rejected. Replacement is atomic for
readers but does not promise durability across a power failure.

Export preserves map definitions, including kernel perf, GPU, and custom map
attributes. Exporting metadata does not require allocating the corresponding
resources and does not guarantee that this build can import those definitions.

Transactional import support depends on the build. Built-in userspace maps are
restored; libbpf-enabled builds also accept program arrays, stack-trace maps,
and lazy kernel-user array/hash/per-CPU-array wrappers. Import recreates those
wrappers without opening or validating their referenced kernel maps.
CUDA-enabled builds also restore host-backed GPU arrays
(`GPU_ARRAY_HOST_MAP` and `PERGPUTD_ARRAY_HOST_MAP`) without registering or
mapping GPU memory. Per-thread host arrays require a positive thread count and
a per-entry size that fits the runtime's 32-bit value-size interface.
Callback-only custom map definitions can be restored when the destination's
`allow_non_buildin_map_types` setting is enabled and the importing process has
registered at least one data-operation callback, with no `alloc_map` or
`map_free` callback. Import does not invoke those callbacks or acquire external
resources. Callbacks are process-local and are not serialized; each process
using the imported map must register its own operations.
External maps with lifecycle callbacks, kernel perf transport, and other GPU map
types remain rejected on import: their constructors and cleanup do not yet
provide the required rollback guarantees. This is an import compatibility limitation
compared with the older constructor-based path, not an absence of runtime support for those map types.
Export retains legacy inspection metadata for epoll (its handler type only)
and memfd (its name and flags). It does not save epoll registrations or memfd
contents. As in the older importer, `import` and `load` reject these handler
types; a file containing either is rejected in full, without publishing any
handlers. Export success therefore does not guarantee that the file can be
imported. Links support the legacy/custom attachment type `0` and perf-event
attachment type `41`; unsupported link types are rejected rather than losing
their type-specific fields.

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

### run-on-cuda - Execute a CUDA Program

Available only in CUDA-enabled builds:

```bash
bpftimetool run-on-cuda <program> [count [gridX gridY gridZ blockX blockY blockZ]]
```

The default count is 1. Supply all six dimensions or none; omitted dimensions
retain the runtime's existing launch configuration. Invalid counts and dimensions
are rejected instead of being partially converted with `atoi`.

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
cat debug_state.json | jq 'to_entries[] | select(.value.type=="bpf_prog_handler")'
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

### Recreate Definitions in a Separate Segment

First stop concurrent changes to the source segment, then export its definitions:

```bash
bpftimetool export definitions.json
BPFTIME_GLOBAL_SHM_NAME=bpftime_test_definitions bpftimetool import definitions.json
```

Use a fresh, task-specific segment name. Import rejects descriptor collisions;
it does not replace existing objects. Map contents must be initialized separately.
Do not remove a live shared-memory segment to work around a collision.

### Move Definitions Between Compatible Environments

Copying JSON does not make every object portable. Review module paths, PIDs,
tracepoint IDs, kernel map/BTF IDs, program assumptions, and available custom
attach implementations on the destination. AOT bytes are architecture/backend
specific. The format preserves these values; it does not relocate them for a
different machine or kernel.

## JSON Format

The top level is an object keyed by canonical nonnegative decimal descriptor
strings, not a `handlers` array. This is a minimal program and perf-event link:

```json
{
  "10": {
    "type": "bpf_link_handler",
    "attr": {
      "prog_fd": 20,
      "target_fd": 30,
      "attach_type": 41,
      "flags": 0,
      "attach_cookie": 123
    }
  },
  "20": {
    "type": "bpf_prog_handler",
    "name": "return_zero",
    "attr": {
      "type": 1,
      "cnt": 2,
      "insns": "b7000000000000009500000000000000"
    }
  },
  "30": {
    "type": "bpf_perf_event_handler",
    "enabled": false,
    "attr": {
      "type": 2,
      "data_type": "tracepoint_perf_event_data",
      "pid": -1,
      "tracepoint_id": 1
    }
  }
}
```

The example only defines records; its tracepoint ID must be adapted before use.
Descriptor order does not determine creation order. Duplicate JSON keys,
noncanonical descriptor aliases such as `"01"`, numeric overflow, malformed hex,
missing required fields, and dangling or incorrectly typed references are errors.
Program `cnt` must match the number of encoded eight-byte instructions. Import
does not verify bytecode safety or relocate/validate map references embedded in
instructions; it preserves instruction bytes and explicit descriptor numbers.

### Handler Types and Compatibility

- **bpf_map_handler**: Map name and complete attributes, including optional
  `gpu_thread_count` (legacy default `1024`). Map entries are not serialized.
- **bpf_prog_handler**: Program name, type, instruction count, and hex-encoded
  instructions. Optional `attr.aot_insns` stores hex-encoded AOT bytes.
- **bpf_perf_event_handler**: Uprobe/uretprobe, override/replace, kprobe/kretprobe,
  tracepoint, software perf, and custom event definitions. Probe offsets and
  reference-counter offsets retain their full width. `enabled: false` remains
  false even for event constructors that normally start enabled. Custom payloads
  use `attr.data_type: "custom_perf_event_data"` and `attr.attach_argument`;
  the explicit tag permits custom numeric event types to overlap built-in IDs.
  Missing tags retain legacy inference for built-in event types only.
- **bpf_link_handler**: Program/target references, attachment type, flags, and
  optional `attr.attach_cookie`. An absent/null cookie is distinct from an
  explicitly supplied zero cookie.

The descriptor-keyed format remains accepted subject to the import capability
limits above. Legacy missing link
`attach_type` and `flags` default to `0`; missing cookies remain absent. Legacy
program `attach_fds` is converted to link records after reserving all explicit
descriptors, without duplicating an explicit connection. Synthesized link
IDs start at `3`, avoiding the standard handler slots; explicitly
specified descriptor numbers remain unchanged. Synthesized IDs are not reserved
OS descriptors. They can still collide with unrelated process-local descriptors,
so this is not equivalent to the older fake-fd allocation path and does not
resolve the runtime's general shared-ID/process-fd ownership problem.
Missing `attach_fds` means no
legacy attachments. Empty exports are `{}`; legacy `null` is accepted
as an empty graph.

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

- **Export**: Copies and encodes definitions before writing the output file
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
