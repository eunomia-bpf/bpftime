# bpftime daemon: runtime userspace eBPF together with kernel eBPF

The bpftime daemon is a system-level monitoring service that intercepts and redirects eBPF operations from kernel space to userspace, enabling enhanced performance and flexibility for eBPF programs. It traces kernel eBPF syscalls and seamlessly integrates them with bpftime's userspace runtime.

## Overview

The daemon bridges kernel and userspace eBPF by:

- **Transparent Redirection**: Intercepts kernel uprobe/uretprobe eBPF programs and runs them in userspace with up to 10x better performance, without modifying the original programs
- **Shared State**: Enables userspace eBPF programs to share maps with kernel eBPF programs for seamless data exchange
- **Zero-copy Communication**: Uses shared memory architecture for efficient inter-process communication
- **Syscall Tracing**: Monitors BPF-related syscalls (bpf, perf_event_open, ioctl) to track eBPF object lifecycle

## Architecture & Implementation

### Components

The daemon consists of two main parts:

1. **Kernel-space Component** (`kernel/bpf_tracer.bpf.c`):
   - eBPF program that runs in kernel space
   - Traces syscalls: bpf(), perf_event_open(), ioctl(), open/close
   - Captures process lifecycle events (exec/exit)
   - Sends events to userspace via ring buffer

2. **Userspace Component** (`user/`):
   - Event handler that processes kernel events
   - Driver that manages bpftime runtime integration
   - Shared memory manager for cross-process communication

### Key Implementation Details

- **Event Types**: The daemon tracks multiple event types defined in `bpf_tracer_event.h`:
  - `SYS_BPF`: BPF syscall operations (map/program creation)
  - `SYS_PERF_EVENT_OPEN`: Perf event creation (including uprobes)
  - `BPF_PROG_LOAD_EVENT`: eBPF program loading
  - `SYS_IOCTL`: Perf event enable/disable operations
  - `EXEC_EXIT`: Process lifecycle events

- **BPF Program Interception**: When a BPF program is loaded:
  1. Kernel tracer captures the program's instructions and metadata
  2. Daemon relocates map references for userspace execution
  3. Program is registered in bpftime's shared memory runtime
  4. Future uprobe hits are handled by userspace runtime

- **Map Sharing**: Maps created in kernel are mirrored in userspace:
  - Map IDs are tracked in `bpf_obj_id_fd_map`
  - File descriptors are mapped to shared memory handles
  - Both kernel and userspace programs can access the same data

## Usage

### Basic Usage

```console
$ sudo SPDLOG_LEVEL=Debug build/daemon/bpftime_daemon
[2023-10-24 11:07:13.143] [info] Global shm constructed. shm_open_type 0 for bpftime_maps_shm
```

### Command Line Options

```
Usage: bpftime_daemon [OPTION...]
Trace and modify bpf syscalls

  -p, --pid=PID              Process ID to trace
  -u, --uid=UID              User ID to trace  
  -o, --open                 Show open events
  -v, --verbose              Verbose debug output
  -w, --whitelist-uprobe=ADDR
                             Whitelist uprobe function addresses
```

## Run malloc example

```console
$ sudo example/malloc/malloc
libbpf: loading object 'malloc_bpf' from buffer
11:08:11 
11:08:12 
11:08:13 
```

Unlike the kernel malloc without bpftime_daemon, this malloc will not print any message. This is because we modify the load and attach process of bpf and perf event with eBPF in the kernel.

## Trace malloc calls in target

```console
$ sudo SPDLOG_LEVEL=Debug ~/.bpftime/bpftime start example/malloc/victim
malloc called from pid 12314
continue malloc...
malloc called from pid 12314
continue malloc...
malloc called from pid 12314
continue malloc...
malloc called from pid 12314
continue malloc...
malloc called from pid 12314
```

The other console will print the malloc calls in the target process.

```console
20:43:22 
        pid=113413      malloc calls: 9
20:43:23 
        pid=113413      malloc calls: 10
20:43:24 
        pid=113413      malloc calls: 10
20:43:25 
        pid=113413      malloc calls: 10
```

## Advanced Usage

### Environment Variables

- `SPDLOG_LEVEL`: Set logging level (Debug, Info, Warn, Error)
- `BPFTIME_DAEMON_CONFIG`: Path to daemon configuration file (optional)

### Filtering and Whitelisting

The daemon supports filtering eBPF operations:

```bash
# Only trace specific process
sudo bpftime_daemon -p 12345

# Only redirect whitelisted uprobe addresses to userspace
sudo bpftime_daemon -w 0x401234 -w 0x401567

# Trace specific user's processes
sudo bpftime_daemon -u 1000
```

### Integration with bpftime

The daemon automatically integrates with bpftime runtime:

1. **Shared Memory**: Uses boost::interprocess shared memory at `/dev/shm/bpftime_maps_shm`
2. **File Descriptors**: Maps kernel FDs to bpftime handler IDs
3. **Program Loading**: Relocates eBPF instructions for userspace execution
4. **Event Attachment**: Redirects uprobe/uretprobe to Frida-based userspace hooks

## Implementation Notes

### Key Files

- `user/bpf_tracer.cpp`: Main daemon loop and kernel event processing
- `user/handle_bpf_event.cpp`: Event handlers for different syscall types
- `user/bpftime_driver.cpp`: Integration with bpftime runtime
- `kernel/bpf_tracer.bpf.c`: Kernel eBPF program for syscall tracing
- `bpf_tracer_event.h`: Event definitions shared between kernel and userspace

### Security Considerations

- Requires root privileges to load kernel eBPF programs
- Can filter by PID/UID to limit scope
- Whitelist mode restricts which uprobes run in userspace
- Prevents tracing of the daemon itself to avoid recursion

## Debug: use bpftimetool for dump states

The dump result example is in [daemon/test/asserts/malloc.json](test/asserts/malloc.json).

See [tools/bpftimetool/README.md](../tools/bpftimetool/README.md) for how to load and replay it in the kernel.

## Building from Source

The daemon is built as part of the bpftime project:

```bash
# Build with daemon support
make build

# The daemon binary will be at:
# build/daemon/bpftime_daemon
```

## Troubleshooting

1. **Permission Denied**: Ensure running with sudo/root privileges
2. **Shared Memory Issues**: Check `/dev/shm/bpftime_maps_shm` permissions
3. **Missing Uprobes**: Use verbose mode (-v) to see uprobe resolution
4. **Performance**: Adjust duration_ms in config to filter short-lived processes
