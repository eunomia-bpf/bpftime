# bpftime Attach System

The attach system in bpftime provides a modular framework for attaching eBPF programs to various event sources in userspace. It supports multiple attachment types including function probes (uprobes), syscall tracing, and specialized attachments like CUDA kernel instrumentation.

## Architecture Overview

### Core Design Principles

1. **Modular Design**: Each attach type is implemented as a separate module inheriting from `base_attach_impl`
2. **Unified Interface**: All attach implementations follow the same interface pattern for consistency
3. **Extensibility**: New attach types can be easily added by implementing the base interface
4. **Cross-Process Support**: Attach points can be shared across processes using shared memory

### Component Structure

```
attach/
├── base_attach_impl/           # Base interface and common functionality
├── frida_uprobe_attach_impl/   # Function instrumentation using Frida
├── syscall_trace_attach_impl/  # System call tracing
├── nv_attach_impl/             # CUDA/GPU kernel instrumentation
├── simple_attach_impl/         # Simplified attach wrapper
└── text_segment_transformer/   # Binary rewriting for syscall interception
```

## Base Interface

The `base_attach_impl` class defines the core interface that all attach implementations must follow:

```cpp
class base_attach_impl {
public:
    // Detach an attachment by ID
    virtual int detach_by_id(int id) = 0;
    
    // Create attachment with eBPF callback
    virtual int create_attach_with_ebpf_callback(
        ebpf_run_callback &&cb, 
        const attach_private_data &private_data,
        int attach_type) = 0;
    
    // Register custom helper functions
    virtual void register_custom_helpers(
        ebpf_helper_register_callback register_callback);
    
    // Call attach-specific functionality
    virtual void *call_attach_specific_function(
        const std::string &name, void *data);
};
```

### Key Types

- **`ebpf_run_callback`**: Function that executes eBPF program with prepared context
- **`attach_private_data`**: Base class for attach-specific configuration data
- **`override_return_set_callback`**: Thread-local callback for modifying return values

## Attach Implementations

### 1. Frida Uprobe Attach (`frida_uprobe_attach_impl`)

Provides dynamic function instrumentation using Frida-gum framework.

**Features:**
- Function entry probes (uprobe)
- Function return probes (uretprobe)
- Override probes (modify function behavior)
- Function replacement (ureplace)

**Attach Types:**
- `ATTACH_UPROBE` (6): Execute at function entry
- `ATTACH_URETPROBE` (7): Execute at function return
- `ATTACH_UPROBE_OVERRIDE` (1008): Override function behavior
- `ATTACH_UREPLACE` (1009): Replace function entirely

**Key Components:**
- Register state capture via `pt_regs`
- Multiple callback types for different probe scenarios
- Thread-local CPU context management

### 2. Syscall Trace Attach (`syscall_trace_attach_impl`)

Intercepts and traces system calls without kernel support.

**Features:**
- Trace syscall entry and exit
- Per-syscall or global callbacks
- Minimal performance overhead

**Key Components:**
- `trace_event_raw_sys_enter/exit`: eBPF-compatible event structures
- Syscall dispatcher with up to 512 syscall slots
- Integration with text segment transformer for interception

**Workflow:**
1. Text segment transformer hooks syscall instructions
2. Hooks redirect to `dispatch_syscall`
3. Dispatcher calls registered eBPF programs
4. Original syscall executed if not overridden

### 3. CUDA Attach (`nv_attach_impl`)

Instruments CUDA kernels and GPU operations (Linux only).

**Features:**
- Memory operation capture
- Kernel function probes/retprobes
- PTX code transformation
- eBPF-to-PTX compilation

**Attach Types:**
- `ATTACH_CUDA_PROBE` (8): CUDA kernel entry
- `ATTACH_CUDA_RETPROBE` (9): CUDA kernel exit

**Implementation:**
1. Intercepts CUDA runtime APIs
2. Extracts and modifies PTX code
3. Injects eBPF programs as PTX
4. Recompiles and loads modified kernels

### 4. Simple Attach (`simple_attach_impl`)

Wrapper providing simplified attach interface for custom event sources.

**Features:**
- Single callback model
- String-based configuration
- Easy integration for new event types

**Use Case:**
Ideal for prototyping new attach types or simple event sources that don't require complex context preparation.

## Private Data System

Each attach type defines its own `attach_private_data` subclass:

```cpp
struct attach_private_data {
    virtual int initialize_from_string(const std::string_view &sv);
    virtual std::string to_string() const;
};
```

Examples:
- `frida_attach_private_data`: Function address, offset, module info
- `syscall_trace_private_data`: Syscall number, entry/exit flag
- `nv_attach_private_data`: Kernel name, attach type

## Helper Functions

Attach implementations can register custom helper functions:

```cpp
// Global helpers available to all attach types
bpftime_set_retval()      // Set function return value
bpftime_override_return()  // Override with specific value

// Attach-specific helpers registered via register_custom_helpers()
```

## Thread Safety

- Each attach implementation manages its own synchronization
- Thread-local storage for per-thread state (e.g., override callbacks)
- Shared memory operations are atomic where necessary

## Adding New Attach Types

1. Create new directory under `attach/`
2. Inherit from `base_attach_impl`
3. Implement required virtual methods
4. Define custom `attach_private_data` if needed
5. Add to `CMakeLists.txt`
6. Register attach type IDs in runtime

## Platform Support

- **Linux**: All attach types supported
- **macOS**: Limited to Frida-based attachments
- **Windows**: Not currently supported

Special requirements:
- CUDA attach requires NVIDIA GPU and CUDA toolkit
- Syscall trace requires text segment modification permissions
- Some features may require root/elevated privileges

## Performance Considerations

1. **Frida Uprobe**: ~10-100ns overhead per probe
2. **Syscall Trace**: Minimal overhead for untraced syscalls
3. **CUDA Attach**: Overhead varies with kernel complexity
4. **Simple Attach**: Depends on callback implementation

## Integration with Runtime

The attach system integrates with the bpftime runtime through:
- Handler manager for object lifecycle
- Shared memory for cross-process access
- VM interface for eBPF program execution
- Helper function registration system