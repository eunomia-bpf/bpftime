# opensnoop

Here is a example that demonstrates the usage of userspace syscall trace

## Usage

Build example, see [documents/build-and-test.md](../../documents/build-and-test.md) for how to build the project.

Build the example:

```sh
make
```

Start server:

```sh
$ sudo ~/.bpftime/bpftime load benchmark/hash_maps/opensnoop
[2023-10-01 16:46:43.409] [info] manager constructed
[2023-10-01 16:46:43.409] [info] global_shm_open_type 0 for bpftime_maps_shm
[2023-10-01 16:46:43.410] [info] Closing 3
[2023-10-01 16:46:43.411] [info] mmap64 0
[2023-10-01 16:46:43.411] [info] Calling mocked mmap64
[2023-10-01 16:46:43.411] [info] Closing 3
[2023-10-01 16:46:43.411] [info] Closing 3
[2023-10-01 16:46:43.423] [info] Closing 3
[2023-10-01 16:46:43.423] [info] Closing 3
```

Start victim:

```console
$ sudo ~/.bpftime/bpftime start -s benchmark/hash_maps/victim
[2023-10-01 16:46:58.855] [info] Entering new main..
[2023-10-01 16:46:58.855] [info] Using agent /root/.bpftime/libbpftime-agent.so
[2023-10-01 16:46:58.856] [info] Page zero setted up..
[2023-10-01 16:46:58.856] [info] Rewriting segment from 559a839b4000 to 559a839b5000
[2023-10-01 16:46:58.859] [info] Rewriting segment from 7f130aa22000 to 7f130ab9a000
[2023-10-01 16:46:59.749] [info] Rewriting segment from 7f130acc3000 to 7f130adb0000
[2023-10-01 16:47:00.342] [info] Rewriting segment from 7f130ae9c000 to 7f130afcd000
[2023-10-01 16:47:01.072] [info] Rewriting segment from 7f130b125000 to 7f130b1a3000
.....
[2023-10-01 16:47:02.084] [info] Attach successfully
[2023-10-01 16:47:02.084] [info] Transformer exiting..

Opening test.txt..
VICTIM: get fd 3
VICTIM: closing fd
Opening test.txt..
VICTIM: get fd 3
VICTIM: closing f
```

## Basic benchmark

Measures average time usage when calling open(2)

### Without bpf

```console
Average open time usage 24216.86932ns,  count 176
```

### Use kernel bpf

```console
Average open time usage 29619.08015ns,  count 262
```

### Use userspace syscall trace (ubpf)

```console
Average open time usage 85299.80000ns, count 153
```

### Use userspace syscall trace (llvm-jit)

```console
Average open time usage 25202.18121ns,  count 95
```
