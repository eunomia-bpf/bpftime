# tailcall example

This is an example demonstrating userspace to kernel tailcall.

## Usage

```
make -j$(nproc)
```

### Terminal 1
```
bpftime load ./tailcall_minimal
```

### Terminal 2

```
bpftime start ./victim
```

### Behavior

See `/sys/kernel/debug/tracing/trace_pipe ` and check whether there are lines containing `Invoked!`

## About this example

`tailcall_minimal.bpf.c` itself is a uprobe ebpf program. It will be triggered when `./victim:add_func` was invoked, and it will run `bpf_tail_call` to call a kernel ebpf program, whose kernel fd is stored in the prog array.

`tailcall_minimal.c` is a loader program. It executes the original syscall, loads an ebpf program into kernel (which printf `Invoked!` to trace_pipe), and stores kernel fd into the userspace prog array
