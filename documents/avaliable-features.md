# available kernel features in userspace

## avalibale map types

Userspace eBPF shared memory map types:

- BPF_MAP_TYPE_HASH
- BPF_MAP_TYPE_ARRAY
- BPF_MAP_TYPE_RINGBUF
- BPF_MAP_TYPE_PERF_EVENT_ARRAY
- BPF_MAP_TYPE_PERCPU_ARRAY
- BPF_MAP_TYPE_PERCPU_HASH

User-kernel shared maps:

- BPF_MAP_TYPE_HASH
- BPF_MAP_TYPE_ARRAY
- BPF_MAP_TYPE_PERCPU_ARRAY
- BPF_MAP_TYPE_PERF_EVENT_ARRAY

## avaliable program types

- tracepoint:raw_syscalls:sys_enter
- tracepoint:syscalls:sys_exit_*
- tracepoint:syscalls:sys_enter_*
- uretprobe:*
- uprobe:*

## available helpers

### maps

- `bpf_map_lookup_elem`: Helper function for looking up an element in a BPF map.
- `bpf_map_update_elem`: Helper function for updating an element in a BPF map.
- `bpf_map_delete_elem`: Helper function for deleting an element from a BPF map.

### kernel_helper_group

- `bpf_probe_read`: Helper function for reading data from a kernel address.
- `bpf_ktime_get_ns`: Helper function for getting the current time in nanoseconds.
- `bpf_trace_printk`: Helper function for printing debug messages from eBPF programs.
- `bpf_get_current_pid_tgid`: Helper function for getting the current PID and TGID (Thread Group ID).
- `bpf_get_current_uid_gid`: Helper function for getting the current UID (User ID) and GID (Group ID).
- `bpf_get_current_comm`: Helper function for getting the current process's name.
- `bpf_strncmp`: Helper function for comparing two strings.
- `bpf_get_func_arg`: Helper function for getting the value of a function argument.
- `bpf_get_func_ret`: Helper function for getting the value of a function return ID.
- `bpf_get_retval`: Helper function for getting the return value of a function.
- `bpf_set_retval`: Helper function for setting the return value of a function.
- `bpf_probe_read_str`: Helper function for reading a null-terminated string from a user address.
- `bpf_get_stack`: Helper function for retrieving the current kernel stack.

## Others

- Support kernel or userspace verifier
- Test JIT with bpf_conformance
