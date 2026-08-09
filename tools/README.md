# Tools

This directory contains additional tools for bpftime to improve the user experience, providing high-level interfaces for eBPF program management, compilation, and runtime control.

## Installation

Install all CLI tools and libraries:

```bash
make install
export PATH=$PATH:~/.bpftime
```

After installation, the following tools will be available in `~/.bpftime/`:

- `bpftime` - Main CLI interface for running and managing eBPF programs
- `bpftimetool` - Tool for inspecting and managing shared memory state
- `bpftime-aot` - AOT (Ahead-of-Time) compilation tool for eBPF to native code

## Tools Overview

### 1. bpftime CLI (`bpftime`)

**Purpose**: High-level interface for injecting bpftime runtime into target processes using `LD_PRELOAD`. This is the primary user-facing tool for running eBPF programs in userspace.

**Usage**:
```bash
bpftime [OPTIONS] COMMAND
```

**Commands**:

#### `load` - Start application with syscall server
Injects the bpftime syscall server into the target application to intercept eBPF-related system calls.

```bash
bpftime load <COMMAND> [ARGS...]
```

**Example**:
```bash
# Run application with eBPF syscall interception
bpftime load ./my_ebpf_program
```

#### `start` - Start application with bpftime agent
Injects the bpftime agent into the target application for eBPF program execution.

```bash
bpftime start [OPTIONS] <COMMAND> [ARGS...]
```

**Options**:
- `-s, --enable-syscall-trace`: Enable syscall tracing functionality

**Examples**:
```bash
# Basic agent injection
bpftime start ./target_application

# With syscall tracing enabled
bpftime start -s ./target_application
```

#### `attach` - Inject bpftime agent to running process
Dynamically attaches the bpftime agent to an already running process using Frida injection.

```bash
bpftime attach [OPTIONS] <PID>
```

**Options**:
- `-s, --enable-syscall-trace`: Enable syscall tracing functionality

**Examples**:
```bash
# Attach to process with PID 1234
bpftime attach 1234

# Attach with syscall tracing
bpftime attach -s 1234
```

#### `detach` - Detach all attached agents
Sends SIGUSR1 signal to all processes with attached bpftime agents to trigger detachment.

```bash
bpftime detach
```

**Global Options**:
- `-i, --install-location <PATH>`: Specify bpftime installation directory (default: `~/.bpftime`)
- `-d, --dry-run`: Run without committing any modifications

### 2. bpftime Tool (`bpftimetool`)

**Purpose**: Inspect and manage the shared memory state containing eBPF objects (programs, maps, links). Provides serialization capabilities for persistent storage and transfer.

**Usage**:
```bash
bpftimetool COMMAND [OPTIONS]
```

**Commands**:

#### `export` - Export shared memory to JSON
Serializes the current bpftime shared memory state to a JSON file for backup or analysis.

```bash
bpftimetool export <filename.json>
```

**Example**:
```bash
bpftimetool export my_ebpf_state.json
```

#### `import` - Import JSON into shared memory
Loads eBPF objects from a JSON file into the bpftime shared memory system.

```bash
bpftimetool import <filename.json>
```

**Example**:
```bash
bpftimetool import my_ebpf_state.json
```

#### `load` - Load JSON with specific fd mapping
Loads eBPF objects from JSON with a specific file descriptor mapping.

```bash
bpftimetool load <fd> <JSON>
```

**Parameters**:
- `<fd>`: File descriptor to map
- `<JSON>`: JSON string containing eBPF objects

**Note**: This command is primarily for internal use when mapping specific file descriptors.

#### `remove` - Remove global shared memory
Completely removes the bpftime shared memory system wide.

```bash
bpftimetool remove
```

#### `run` - Execute eBPF program from shared memory
Runs an eBPF program stored in shared memory with provided input data.

```bash
bpftimetool run <id> <data_file> [repeat <N>] [type <RUN_TYPE>]
```

**Parameters**:
- `<id>`: Program ID in shared memory
- `<data_file>`: Binary file containing input data for the program
- `repeat <N>`: Number of times to execute (default: 1)
- `type <RUN_TYPE>`: Execution mode (JIT, AOT, INTERPRET)

**Run Types**:
- `JIT`: Just-In-Time compilation (default)
- `AOT`: Ahead-of-Time compiled code
- `INTERPRET`: Interpreted execution

**Examples**:
```bash
# Run program ID 0 with input data
bpftimetool run 0 input.bin

# Run with 1000 iterations for benchmarking
bpftimetool run 0 input.bin repeat 1000

# Run AOT-compiled version
bpftimetool run 0 input.bin type AOT

# Combined options
bpftimetool run 0 input.bin repeat 100 type JIT
```

### 3. bpftime AOT Tool (`bpftime-aot`)

**Purpose**: Ahead-of-Time (AOT) compilation tool that compiles eBPF bytecode to native machine code for maximum performance. Supports both ELF-based compilation and shared memory compilation.

**Usage**:
```bash
bpftime-aot COMMAND [OPTIONS]
```

**Commands**:

#### `build` - Compile eBPF ELF to native ELF
Compiles eBPF programs from an ELF file to native machine code.

```bash
bpftime-aot build [OPTIONS] <EBPF_ELF>
```

**Options**:
- `-o, --output <DIR>`: Output directory (default: current directory)
- `-e, --emit_llvm`: Emit LLVM IR files alongside native code

**Example**:
```bash
# Compile eBPF ELF to native code
bpftime-aot build my_program.bpf.o -o ./output/

# Emit LLVM IR for analysis
bpftime-aot build my_program.bpf.o -e
```

#### `compile` - Compile programs from shared memory
Compiles eBPF programs currently loaded in bpftime shared memory to native code.

```bash
bpftime-aot compile [OPTIONS]
```

**Options**:
- `-o, --output <DIR>`: Output directory (default: current directory)
- `-e, --emit_llvm`: Emit LLVM IR files alongside native code

**Example**:
```bash
# Compile all programs in shared memory
bpftime-aot compile -o ./compiled/

# With LLVM IR emission
bpftime-aot compile -e
```

#### `load` - Load AOT object into shared memory
Loads a pre-compiled native ELF file into shared memory for a specific program ID.

```bash
bpftime-aot load <ELF_PATH> <PROGRAM_ID>
```

**Example**:
```bash
# Load compiled object for program ID 0
bpftime-aot load do_count.o 0
```

#### `run` - Execute native eBPF program
Directly executes a native eBPF program with optional input data.

```bash
bpftime-aot run <ELF_PATH> [MEMORY_FILE]
```

**Parameters**:
- `<ELF_PATH>`: Path to native ELF file
- `[MEMORY_FILE]`: Optional binary file with input memory

**Example**:
```bash
# Run native program
bpftime-aot run do_count.o

# Run with input data
bpftime-aot run do_count.o input_data.bin
```

### AOT Compilation Workflow

The AOT tool supports two main workflows:

#### 1. Direct ELF Compilation
```bash
# Compile eBPF ELF to native
bpftime-aot build program.bpf.o -o output/

# Link with your application
clang -O2 main.c output/program.o -o final_app
```

#### 2. Shared Memory Compilation
```bash
# Load eBPF program (using syscall server)
LD_PRELOAD=libbpftime-syscall-server.so ./ebpf_loader

# Compile loaded programs
bpftime-aot compile -o output/

# Load back into shared memory
bpftime-aot load output/program.o 0
```

## Development and Testing Tools

### ARM64 Build Testing
Test bpftime compilation on ARM64 architecture using Docker:

```bash
./tools/test_arm_build.sh
```

This script:
1. Builds Docker image for ARM64
2. Tests release build compilation
3. Verifies example programs build correctly

### Docker Support
Multiple Dockerfiles for different environments:

- `Dockerfile.ubuntu`: Ubuntu 22.04 based development environment
- `Dockerfile.fedora`: Fedora based environment  
- `Dockerfile.arm`: ARM64 cross-compilation environment

## Integration Examples

### Complete Workflow Example

1. **Load eBPF program with syscall interception**:
   ```bash
   bpftime load ./my_ebpf_app
   ```

2. **Export state for analysis**:
   ```bash
   bpftimetool export program_state.json
   ```

3. **Compile to native code**:
   ```bash
   bpftime-aot compile -o native_output/
   ```

4. **Benchmark different execution modes**:
   ```bash
   bpftimetool run 0 test_data.bin repeat 1000 type INTERPRET
   bpftimetool run 0 test_data.bin repeat 1000 type JIT
   bpftimetool run 0 test_data.bin repeat 1000 type AOT
   ```

### Performance Testing
```bash
# Load and run with performance measurement
bpftime start ./target_app
bpftimetool run 0 input.bin repeat 10000 type JIT
```

## Environment Variables

### Common Variables
- `SPDLOG_LEVEL`: Control logging level (trace, debug, info, warn, error, critical)
  ```bash
  SPDLOG_LEVEL=debug bpftimetool import program.json
  ```

### bpftime CLI Variables
- `HOME`: Used to determine default installation location (`~/.bpftime`)
- `PATH`: Should include bpftime installation directory after installation
- `LD_PRELOAD`: Automatically set by the tool when injecting libraries
- `AGENT_SO`: Used internally to specify the agent library path

### Platform-Specific Libraries
The tools automatically use the correct library names based on platform:
- **Linux**: `.so` files (e.g., `libbpftime-agent.so`)
- **macOS**: `.dylib` files (e.g., `libbpftime-agent.dylib`)
