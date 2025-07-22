# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

bpftime is a high-performance userspace eBPF runtime and general extension framework. It enables running eBPF programs in userspace for observability, networking, GPU, and other use cases, achieving up to 10x better performance than kernel eBPF for certain operations.

## Common Development Commands

### Build Commands

```bash
# Build with tests and all components (Debug mode)
make build

# Build release version
make release

# Build with LLVM JIT support
make release-with-llvm-jit

# Build with static library
make release-with-static-lib

# Clean build artifacts
make clean

# Build specific components
cmake --build build --target <target_name>
```

### Testing Commands

```bash
# Run all unit tests
make unit-test

# Run runtime tests only
make unit-test-runtime

# Run daemon tests only
make unit-test-daemon

# Build eBPF test programs (from runtime/test/bpf)
make -C runtime/test/bpf

# Run specific test binary
./build/runtime/unit-test/bpftime_runtime_tests
./build/daemon/test/bpftime_daemon_tests
```

### Running Examples

```bash
# Build example programs
make -C example/malloc

# Load and run eBPF program
export PATH=$PATH:~/.bpftime/
bpftime load ./example/malloc/malloc
bpftime start ./example/malloc/victim

# Attach to running process (requires sudo)
sudo bpftime attach <pid>
```

### Development Tools

```bash
# Install to system (after build)
sudo cmake --install build

# Build with specific features
cmake -Bbuild -DBPFTIME_ENABLE_IOURING_EXT=1  # IO uring support
cmake -Bbuild -DBPFTIME_LLVM_JIT=1           # LLVM JIT backend
cmake -Bbuild -DBPFTIME_ENABLE_MPK=1         # Memory Protection Keys
```

## Architecture & Code Structure

### Component Organization

The codebase is organized into distinct components that interact through well-defined interfaces:

1. **`vm/`** - Virtual Machine implementations
   - `vm-core/`: Core VM interface and abstractions
   - `llvm-jit/`: LLVM-based JIT/AOT compiler (high performance)
   - Supports multiple backends (LLVM JIT, ubpf)

2. **`runtime/`** - Core runtime functionality
   - `include/`: Public APIs and interfaces
   - `src/handler/`: Handler pattern for managing eBPF objects (programs, maps, links)
   - `src/bpf_map/`: Map implementations (hash, array, ringbuf, etc.)
   - `syscall-server/`: Intercepts eBPF syscalls for compatibility
   - `agent/`: Library injected into target processes

3. **`attach/`** - Event attachment mechanisms
   - `base_attach_impl/`: Abstract interface for all attach types
   - `frida_uprobe_attach_impl/`: Dynamic instrumentation via Frida
   - `syscall_trace_attach_impl/`: Syscall interception
   - `nv_attach_impl/`: CUDA/GPU event attachment

4. **`daemon/`** - System monitoring and kernel interaction
   - Monitors kernel eBPF operations
   - Can redirect to userspace runtime

5. **`bpftime-verifier/`** - Safety verification
   - Wraps PREVAIL verifier for userspace
   - Optional kernel verifier integration

### Key Architectural Patterns

1. **Shared Memory Architecture**: Uses boost::interprocess for zero-copy IPC between processes
   - Central `handler_manager` registry in shared memory
   - File descriptor abstraction for kernel compatibility

2. **Handler Pattern**: All eBPF objects (programs, maps, links) managed through handlers
   - Consistent interface for create/destroy/access
   - Enables cross-process object sharing

3. **Modular Attach System**: New event sources added by implementing `base_attach_impl`
   - Each attach type provides its own helpers
   - Supports custom context preparation

4. **VM Abstraction**: Multiple VM backends through common interface
   - Switch between interpreter/JIT/AOT transparently
   - Extensible helper function registration

### Important Files and Interfaces

- `runtime/include/bpftime.hpp`: Main runtime API
- `runtime/include/bpftime_shm.hpp`: Shared memory management
- `vm/vm-core/include/ebpf-vm.h`: VM interface
- `attach/base_attach_impl/base_attach_impl.hpp`: Attach interface
- `runtime/src/handler/handler_manager.hpp`: Central object registry

### Development Workflow

1. **Adding new functionality**: Check existing patterns in similar components
2. **Testing**: Add unit tests in component's test directory
3. **eBPF programs**: Use standard clang/libbpf toolchain, test with both userspace and kernel
4. **Performance**: Consider shared memory overhead, prefer batch operations

The architecture prioritizes performance (bypassing kernel), compatibility (same APIs as kernel eBPF), and extensibility (modular design for new features).