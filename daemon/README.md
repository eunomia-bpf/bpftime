# bpftime daemon: trace and replay eBPF related events

The bpftime daemon is a tool to trace and replay eBPF related events.
It's similar to our syscall server but run together with kernel eBPF.

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

## Debug: use bpftimetool for dump states

The dump result example is in [daemon/test/malloc.json](test/malloc.json).

See [tools/bpftimetool/README.md](../tools/bpftimetool/README.md) for how to load and replay it in the kernel.

