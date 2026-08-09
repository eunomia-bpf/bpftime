# bpftime-aot

Ahead-of-Time (AOT) compiler for eBPF programs that converts eBPF bytecode to native machine code using LLVM.

## Overview

`bpftime-aot` is a command-line tool that compiles eBPF programs to native ELF objects for high-performance execution in userspace. It supports:

- **AOT compilation** from eBPF bytecode to native x86/ARM machine code
- **Multiple input sources**: eBPF ELF files or programs in shared memory
- **Standalone execution** of compiled programs
- **Helper function relocation** for seamless integration
- **LLVM IR emission** for debugging and optimization analysis

For the underlying library, see [llvmbpf](https://github.com/eunomia-bpf/llvmbpf).

## Installation

After building bpftime, the tool is available at:
```bash
~/.bpftime/bpftime-aot
# Or add to PATH
export PATH=$PATH:~/.bpftime/
```

## Usage

```console
bpftime-aot [--help] [--version] {build,compile,load,run}

Subcommands:
  build      Build native ELF(s) from eBPF ELF object file
  compile    Compile eBPF programs loaded in bpftime shared memory
  load       Load a compiled native ELF into shared memory
  run        Execute a compiled native ELF program
```

## Command Reference

### build - Compile from eBPF ELF

Compile eBPF programs from an ELF object file to native code:

```bash
bpftime-aot build <EBPF_ELF> [-o OUTPUT_DIR] [-e]

Options:
  -o, --output DIR    Output directory (default: current directory)
  -e, --emit_llvm     Emit LLVM IR instead of native object code
```

**Example:**
```bash
# Compile all programs in an eBPF ELF
bpftime-aot build example/uprobe.bpf.o -o output/

# Generate LLVM IR for analysis
bpftime-aot build example/uprobe.bpf.o -e
```

Each eBPF program in the ELF will produce a separate `.o` file (e.g., `do_uprobe.o`).

### compile - Compile from Shared Memory

Compile eBPF programs already loaded in bpftime shared memory:

```bash
bpftime-aot compile [-o OUTPUT_DIR] [-e]
```

**Example - Full workflow:**

1. Load eBPF programs into shared memory:

```bash
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so example/malloc/malloc
```

2. Compile the loaded program to native code:

```bash
bpftime-aot compile
```

Output: `do_count.o` (native ELF object)

**Key advantage:** When compiling from shared memory, maps, global variables, and helper functions are already relocated, making the compiled code ready for integration.

### load - Load Compiled ELF to Shared Memory

Load a pre-compiled native ELF into shared memory for execution:

```bash
bpftime-aot load <PATH> <ID>

Arguments:
  PATH    Path to the compiled native ELF file
  ID      Program ID in shared memory to update
```

**Example:**
```bash
# Load compiled native code for program ID 4
bpftime-aot load do_count.o 4
```

### run - Execute Compiled Program

Run a compiled native ELF program directly:

```bash
bpftime-aot run <PATH> [MEMORY]

Arguments:
  PATH      Path to the compiled native ELF
  MEMORY    Optional: Path to memory file for program context
```

**Example:**
```bash
# Run compiled program
bpftime-aot run do_uprobe_trace.o

# Run with memory context
bpftime-aot run program.o memory.bin
```

## Integration Examples

### Linking with Custom Programs

You can link compiled eBPF programs with your C/C++ applications:

```bash
cd tools/aot/example
clang -O2 main.c do_count.o -o malloc
./malloc
```

The driver program needs to implement required helper functions:

```c
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <stdarg.h>

// Entry point - called from main
int bpf_main(void* ctx, uint64_t size);

// Helper function implementations
uint64_t _bpf_helper_ext_0006(uint64_t fmt, uint64_t fmt_size, ...) {
    // bpf_printk implementation
    va_list args;
    va_start(args, fmt);
    vprintf((const char *)fmt, args);
    va_end(args);
    return 0;
}

uint64_t _bpf_helper_ext_0014(void) {
    // bpf_get_current_pid_tgid implementation
    return ((uint64_t)getpid() << 32) | gettid();
}

// Map operations (simplified mock)
uint64_t counter = 0;

void *_bpf_helper_ext_0001(void *map, const void *key) {
    return &counter;  // bpf_map_lookup_elem
}

long _bpf_helper_ext_0002(void *map, const void *key,
                          const void *value, uint64_t flags) {
    counter = *(uint64_t*)value;  // bpf_map_update_elem
    return 0;
}

uint64_t __lddw_helper_map_by_fd(uint32_t id) {
    return 0;  // Map relocation helper
}

int main() {
    bpf_main(NULL, 0);
    return 0;
}
```

### Helper Function Naming Convention

Helper functions are named `_bpf_helper_ext_XXXX` where `XXXX` is the helper ID:
- `0001` = bpf_map_lookup_elem
- `0002` = bpf_map_update_elem
- `0006` = bpf_printk
- `0014` = bpf_get_current_pid_tgid

### Understanding Relocation

When compiling from shared memory (using `bpftime-aot compile`):
- **Maps are relocated**: `__lddw_helper_map_by_fd` receives actual shared memory map IDs
- **Helpers are resolved**: Helper function addresses are fixed up
- **Global variables are linked**: Accessible through shared memory

See `tools/aot/example/` for complete working examples.

## Advanced Usage

### Emitting LLVM IR

Generate LLVM IR for optimization analysis or debugging:

```bash
# From eBPF ELF
bpftime-aot build uprobe.bpf.o -e

# From shared memory
bpftime-aot compile -e
```

Output files will have `.ll` extension containing human-readable LLVM IR.

### Example Output

```console
$ bpftime-aot run do_uprobe_trace.o
[info] [llvm_jit_context.cpp:81] Initializing llvm
[info] [llvm_jit_context.cpp:204] LLVM-JIT: Loading aot object
target_func called.
[info] [main.cpp:190] Output: 0
```

## Performance Benefits

AOT compilation provides significant performance advantages:
- **No JIT overhead**: Compilation happens once, offline
- **Optimized native code**: Full LLVM optimization pipeline
- **Reduced startup time**: Programs start immediately
- **Better code placement**: Improved instruction cache utilization

Typical speedup: 2-5x faster than JIT, 10-50x faster than interpreter.

## Troubleshooting

**Q: "Unable to open BPF elf" error**
A: Ensure the eBPF ELF file is compiled with clang and contains valid BTF information.

**Q: "Invalid id not exist" when loading**
A: Verify the program ID exists in shared memory using `bpftimetool export`.

**Q: Helper function undefined**
A: Implement all required `_bpf_helper_ext_XXXX` functions in your driver program.

## See Also

- [bpftimetool](https://github.com/eunomia-bpf/bpftime/tree/master/tools/bpftimetool) - Shared memory inspection tool
- [optimize.md](https://github.com/eunomia-bpf/bpftime/blob/master/tools/aot/optimize.md) - Optimization guide
- [llvmbpf](https://github.com/eunomia-bpf/llvmbpf) - Underlying LLVM library
- [bpftime documentation](https://github.com/eunomia-bpf/bpftime) - Main project documentation
