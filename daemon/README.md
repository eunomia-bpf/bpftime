# bpftime daemon: runtime userspace eBPF together with kernel eBPF

The bpftime daemon is a tool to trace and run eBPF programs in userspace with kernel eBPF. It can:

- make original kernel uprobe and uretprobe eBPF programs actually running in userspace with userspace advantages. Without any modification, no syscall-server is needed.
- make Userspace eBPF programs share the same maps with kernel eBPF programs.

## Run daemon

```console
$ sudo SPDLOG_LEVEL=Debug build/daemon/bpftime_daemon
[2023-10-24 11:07:13.143] [info] Global shm constructed. shm_open_type 0 for bpftime_maps_shm
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

## Debug: use bpftimetool for dump states

The dump result example is in [daemon/test/asserts/malloc.json](test/asserts/malloc.json).

See [tools/bpftimetool/README.md](../tools/bpftimetool/README.md) for how to load and replay it in the kernel.
