# Syscall Trace Attach Implementation

This component provides a mechanism for attaching eBPF programs to system calls in userspace with bpftime.

## Overview

The syscall trace attach implementation allows intercepting and monitoring system calls by attaching eBPF programs to:
- Specific syscall entry points (e.g., entering `read`, `write`)
- Specific syscall exit points (e.g., exiting `read`, `write`)
- All syscall entries (global enter hook)
- All syscall exits (global exit hook)

## Core Components

- **Syscall Table**: Provides mappings between syscall IDs, names, and tracepoints
- **Syscall Trace Attach Implementation**: Core implementation for attaching eBPF callbacks to syscalls
- **Private Data**: Configuration structures for the attach operation

## How It Works

1. **Syscall Mapping**: System calls are mapped from their IDs to names and tracepoints using `/sys/kernel/tracing/events/syscalls/*`
2. **Attachment**: eBPF programs can be attached to specific syscall entry/exit points through the `syscall_trace_attach_impl` class
3. **Dispatch**: When a syscall is intercepted, the implementation dispatches the call to the appropriate registered callbacks
4. **Callbacks**: Registered callbacks receive context information including syscall arguments and can optionally override return values

## Usage Example

To attach an eBPF program to the `read` syscall entry:

```cpp
// Create the attach implementation
syscall_trace_attach_impl attacher;

// Create private data for the "read" syscall entry
syscall_trace_attach_private_data data;
data.initialize_from_string("..."); // Tracepoint ID string for sys_enter_read

// Attach an eBPF callback
int attach_id = attacher.create_attach_with_ebpf_callback(
    [](const void *ctx, size_t size, uint64_t *ret) -> int {
        // Access syscall context
        auto &enter_ctx = *(trace_event_raw_sys_enter *)ctx;
        // Process syscall arguments
        // enter_ctx.args[0], enter_ctx.args[1], etc.
        return 0;
    },
    data,
    ATTACH_SYSCALL_TRACE
);

// Later, detach by ID
attacher.detach_by_id(attach_id);
```

## Integration with Text Segment Transformer

This implementation is designed to work with the Text Segment Transformer, which modifies the executable code of a process to intercept syscalls at runtime.

The syscall tracing implementation receives syscall events from the transformer and dispatches them to the registered eBPF programs. 