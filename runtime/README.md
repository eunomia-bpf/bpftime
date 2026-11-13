# bpftime Runtime

The bpftime runtime is the core component that provides a high-performance userspace eBPF execution environment. It enables running eBPF programs in userspace with up to 10x better performance than kernel eBPF for certain operations, while maintaining compatibility with existing eBPF toolchains.

## Architecture Overview

The runtime implements a sophisticated shared memory architecture using Boost.Interprocess to enable zero-copy IPC between processes. At its core is a handler pattern that manages all eBPF objects (programs, maps, links) through a consistent interface.

### Key Architectural Components

1. **Shared Memory Management** (`bpftime_shm.cpp`, `bpftime_shm_internal.cpp`)
   - Central registry in shared memory for cross-process object sharing
   - File descriptor abstraction maintaining kernel compatibility
   - JSON import/export for persistence and debugging

2. **Handler System** (`src/handler/`)
   - `handler_manager`: Central registry managing all eBPF objects
   - Type-specific handlers: `prog_handler`, `map_handler`, `link_handler`, `perf_event_handler`
   - Uses `std::variant` for type-safe polymorphic storage

3. **VM Integration** (`bpftime_prog.cpp`)
   - Abstracts multiple VM backends (LLVM JIT, ubpf interpreter)
   - Helper function registration and management
   - Cross-platform execution support

## Implementation Details

### Core APIs (`include/bpftime.hpp`)

The main runtime API provides C interfaces for:
- Program lifecycle: `bpftime_progs_create()`, program execution
- Map operations: `bpftime_maps_create()`, lookup/update/delete
- Event attachment: `bpftime_uprobe_create()`, `bpftime_link_create()`
- Shared memory: `bpftime_initialize_global_shm()`, JSON import/export

### BPF Maps Implementation (`src/bpf_map/`)

The runtime provides a comprehensive set of BPF map types organized into three categories:

#### Userspace Maps (`userspace/`)
Pure userspace implementations optimized for performance:
- **Basic maps**: array, hash (fixed & variable size)
- **Per-CPU maps**: per-cpu array/hash with CPU affinity
- **Specialized structures**: ringbuf, queue, stack, bloom filter
- **Advanced types**: LPM trie, LRU hash, prog array, stack trace
- **Map-in-maps**: array of maps

Key features:
- Zero-copy operations via shared memory
- Custom optimizations (fixed-size hash tables, per-CPU structures)
- Extended operations (push/pop/peek for queues/stacks)

#### Shared Maps (`shared/`)
Bridge between kernel and userspace eBPF programs:
- Array, hash, per-CPU array, perf event array
- Delegate operations to kernel via BPF syscalls
- Enable data sharing between kernel and userspace contexts

#### GPU Maps (`gpu/`)
CUDA-accelerated maps for GPU workloads:
- GPU array and ringbuf implementations
- Cross-process GPU memory sharing via `CUipcMemHandle`
- Per-thread GPU buffer support

### Handler Implementation Pattern

All handlers follow a consistent pattern:

```cpp
class handler {
    // Core operations
    int create(args...);
    void destroy();
    
    // Type-specific operations
    // e.g., for maps: lookup, update, delete
    // e.g., for progs: load, execute
};
```

Handlers are stored in a global `handler_variant_vector` in shared memory, indexed by file descriptor for kernel compatibility.

### Attach Mechanisms (`src/attach/`)

The runtime supports various eBPF program attachment types:
- **Uprobe/uretprobe**: Function entry/exit hooking via Frida
- **Syscall tracing**: System call interception
- **Custom events**: Extensible for new event sources (e.g., CUDA)

### Agent and Syscall Server

#### Agent (`agent/agent.cpp`)
- Library injected into target processes
- Intercepts eBPF-related system calls
- Redirects to userspace runtime or kernel as configured

#### Syscall Server (`syscall-server/`)
- Standalone process managing eBPF operations
- Handles syscall redirection from agents
- Provides compatibility layer for existing eBPF tools

## Memory Model

### Shared Memory Layout
```
Global Shared Memory (bpftime_maps_shm)
├── handler_manager (central registry)
│   ├── prog_handlers
│   ├── map_handlers
│   ├── link_handlers
│   └── perf_event_handlers
├── Map data (varies by type)
├── Program bytecode
└── Runtime configuration
```

### Synchronization
- Boost interprocess mutexes for cross-process synchronization
- pthread spinlocks for high-performance map operations
- Lock-free designs where possible (e.g., simple array maps)

## Extension Points

### Adding New Map Types
1. Implement the map interface in `src/bpf_map/userspace/`
2. Register in `bpf_map_handler::create_map_impl()`
3. Add type enum in `bpf_map_type`

### Adding New Attach Types
1. Implement attach mechanism in `attach/` directory
2. Register in attach context
3. Add helper functions if needed

### External Map Operations
Use `bpftime_register_map_ops()` to register custom map implementations without modifying core runtime.

## Build System

The runtime uses CMake with several build options:
- `BPFTIME_LLVM_JIT`: Enable LLVM JIT backend
- `BPFTIME_ENABLE_IOURING_EXT`: IO uring support
- `BPFTIME_ENABLE_MPK`: Memory Protection Keys
- `BPFTIME_BUILD_WITH_LIBBPF`: Kernel map sharing

## Testing

### Unit Tests (`unit-test/`)
- Comprehensive tests for maps, programs, attachments
- Shared memory operation tests
- Cross-platform compatibility tests

### Integration Tests (`test/`)
- End-to-end eBPF program execution
- Multi-process shared memory tests
- Performance benchmarks

## Maintenance Guide

### Common Issues and Solutions

1. **Shared Memory Corruption**
   - Check `BPFTIME_GLOBAL_SHM_NAME` environment variable
   - Use `bpftime_remove_global_shm()` to clean up
   - Export/import JSON for debugging

2. **Map Operation Failures**
   - Verify map type compatibility
   - Check `should_lock` flag for thread safety
   - Ensure proper initialization order

3. **Program Loading Issues**
   - Validate bytecode with verifier
   - Check helper function registration
   - Verify VM backend availability

### Performance Optimization

1. **Map Selection**
   - Use fixed-size maps when possible
   - Consider per-CPU maps for high-contention scenarios
   - Leverage GPU maps for CUDA workloads

2. **VM Backend**
   - LLVM JIT for production performance
   - Interpreter for debugging/development

3. **Shared Memory**
   - Tune allocation sizes in `bpftime_shm_internal.hpp`
   - Use memory protection (MPK) for security

### Debugging Tools

1. **JSON Export/Import**
   
   See the bpftimetool usage in the tool dir.

2. **Logging**
   - Set `SPDLOG_LEVEL=debug` for detailed logs
   - Check handler allocation/deallocation
   - Monitor shared memory usage

3. **Tracing**
   - Use system tracing tools on agent library
   - Monitor syscall interception
   - Profile map operations

## Host memory access safety (probe read/write backends)

All BPF helpers that dereference a host-supplied pointer converge on a single
choke point, `bpftime_probe_read` / `bpftime_probe_write` in
`runtime/src/bpf_helper.cpp`. Unlike kernel eBPF, the userspace runtime has no
hardware `copy_from_user_nofault` boundary: the target address is directly
addressable, so an invalid pointer raises a regular `SIGSEGV` in the host
process. The backend selected at this choke point is therefore a host-safety
boundary, not an implementation detail.

### Backends

| Backend | Build selection | Fault behavior | Cost class |
| --- | --- | --- | --- |
| Unprotected `memcpy` | `ENABLE_PROBE_READ_CHECK=OFF` / `ENABLE_PROBE_WRITE_CHECK=OFF` | Host `SIGSEGV` on any invalid pointer | Plain memory copy, no syscalls |
| No-fault syscall copy | `ENABLE_PROBE_READ_CHECK=ON` (default), write check likewise ON | `-EFAULT`, host survives | One `process_vm_readv`/`process_vm_writev` per call |

A scoped signal-recovery backend (SIGSEGV caught only inside a protected-copy
window, returning `-EFAULT` via `siglongjmp`) and a bounded stack snapshot
backend (one fault-safe capture per BPF event with integer-bounds replay) are
under review. Both must satisfy the semantics below before becoming selectable.

### Required semantics for every safe backend

1. Invalid, unmapped, stale-mapped, and non-canonical addresses return
   `-EFAULT`; the host process and its other threads continue.
2. Host signal dispositions are preserved. A signal-recovery backend may
   install a handler only while a thread is inside a protected-copy window
   marked in thread-local state; any fault outside the window is delivered to
   the previous handler instead of being swallowed. A helper must never leave
   a process-wide handler installed after returning.
3. Small-stack environments (coroutines/fibers using small alternate stacks)
   require an explicit `sigaltstack` so the signal frame can be delivered;
   otherwise the kernel kills the thread before the handler runs.
4. Write-side checks follow the same rules via `process_vm_writev`; an
   unchecked write path is the same explicit opt-in category as unchecked
   reads.
5. When capture or initialization fails, the helper degrades to the
   compile-time safe backend or returns `-EFAULT`; it never falls back to
   unchecked `memcpy`.

### Selection and default policy

- Shipping builds keep a safe backend by default. The unchecked `memcpy`
  backend is reserved for benchmark configurations and must be a deliberate
  configure-time choice.
- The no-fault syscall backend is the conservative default: it takes no
  dependency on Frida and does not alter process signal handling.
- Frame-pointer-only pre-checks (PC LPM trie and frame-pointer walks) are an
  optimization, not a safety mechanism: they do not validate stack or
  frame-pointer addresses, including stale frames left by coroutine stack
  switches. Safety must come from the copy backend.

### Test surface

- `runtime/unit-test/test_probe.cpp`: valid, unmapped, and stale pointers;
  cross-thread `SIGSEGV` delivery; handler replacement and default handler.
- `runtime/unit-test/test_shm_allocation_failure.cpp`: host exit status and
  default stdio silence under allocation failure.
- Signal-recovery backends additionally cover repeated recovery,
  non-canonical addresses, signal-mask restoration, initialization fallback,
  and unrelated `SIGSEGV` retaining fatal default behavior.
