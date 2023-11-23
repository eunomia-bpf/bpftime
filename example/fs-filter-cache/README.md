# fs-cache

Trace file open or close syscalls in a process.

Since the program is not attached to all syscall events, it is not necessary to filter the events for pid and uid in the program. It can also speedup the program and reduce the overhead.

## Usage

run agent

```console
# AGENT_SO=build/runtime/agent/libbpftime-agent.so LD_PRELOAD=build/runtime/agent-transformer/libbpftime-agent-transformer.so find ./example
```

run server:

```console
sudo LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so example/fs-filter-cache/fs-cache
```

## cache for getdents64

passthrough fuse base line of getdents64 syscall

```console
root@yunwei37server:/home/yunwei/bpftime# /home/yunwei/bpftime/example/fs-filter-cache/bench getdents64 /home/yunwei/bpftime-evaluation/fuse/data/virt
inode=4294967295 offset=32 reclen=24 type=0 name=.
inode=4294967295 offset=64 reclen=24 type=0 name=..
inode=4294967295 offset=96 reclen=24 type=0 name=kvm
inode=4294967295 offset=128 reclen=24 type=0 name=lib
inode=4294967295 offset=160 reclen=32 type=0 name=Makefile
Average time usage 36526.149010 ns, iter 100000 times
```

use cache for getdents64 syscall

```console
# AGENT_SO=build/runtime/agent/libbpftime-agent.so LD_PRELOAD=build/runtime/agent-transformer/libbpftime-agent-transformer.so  /home/yunwei/bpftime/example/fs-filter-cache/bench getdents64  /home/yunwei/bpftime-evaluation/fuse/data/virt
....
inode=4294967295 offset=32 reclen=24 type=0 name=.
inode=4294967295 offset=64 reclen=24 type=0 name=..
inode=4294967295 offset=96 reclen=24 type=0 name=kvm
inode=4294967295 offset=128 reclen=24 type=0 name=lib
inode=4294967295 offset=160 reclen=32 type=0 name=Makefile
Average time usage 1761.278320 ns, iter 100000 times

[2023-11-23 23:38:52][info][414216] Compiling using LLJIT
INFO [414216]: Global shm destructed
```
