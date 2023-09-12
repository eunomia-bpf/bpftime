# benchmark of uprobe and uretprobe

With userspace eBPF runntime, we can:

- speed up the uprobe and uretprobe by approximate 10x
- with out any kernel patch or modify the tracing eBPF program
- No privilege is needed for running the eBPF tracing program.

## build

Build the agent first. In project root:

```sh
make build
```

build the benchmark driver:

```sh
make -C benchmark
```

## test environment

```console
$ uname -a
Linux yunwei37server 6.2.0-32-generic #32-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 14 10:03:50 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
```

## base line

```console
$ benchmark/test
a[b] + c for 100000 times
Elapsed time: 0.000446995 seconds
avg function elapse time: 4.469950 ns
```

The base line function elapsed time is 0.000243087 seconds, for the test function:

```c
__attribute_noinline__ 
uint64_t __benchmark_test_function3(const char *a, int b,
          uint64_t c)
{
 return a[b] + c;
}
```

## kernel uprobe

Build the uprobe and uretprobe:

```sh
make -C benchmark/uprobe
make -C benchmark/uretprobe
```

run the uprobe:

```console
$  sudo benchmark/uprobe/uprobe
libbpf: loading object 'uprobe_bpf' from buffer
libbpf: elf: section(2) .symtab, size 120, link 1, flags 0, type=2
...
loaded ebpf program...
...
```

in another terminal, run the benchmark:

```console
$ benchmark/test
a[b] + c for 100000 times
Elapsed time: 0.475146261 seconds
avg function elapse time: 4751.462610 ns
```

The uprobe or uretprobe function we used is like:

```c
SEC("uprobe/benchmark/test:__benchmark_test_function3")
int BPF_UPROBE(__benchmark_test_function, const char *a, int b, uint64_t c)
{
 return b + c;
}
```

## kernel uretuprobe

run the uretprobe:

```console
$  sudo benchmark/uretprobe/uretprobe
libbpf: loading object 'uprobe_bpf' from buffer
libbpf: elf: section(2) .symtab, size 120, link 1, flags 0, type=2
...
loaded ebpf program...
...

in another terminal, run the benchmark:

```console
$ benchmark/test
a[b] + c for 100000 times
Elapsed time: 0.589970682 seconds
avg function elapse time: 5899.706820 ns
```

## userspace uprobe

run the uprobe:

```console
$ LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so benchmark/uprobe/uprobe
manager constructed
global_shm_open_type 0 for bpftime_maps_shm
Closing 3
libbpf: loading object 'uprobe_bpf' from buffer
libbpf: elf: section(2) .symtab, size 120, link 1, flags 0, type=2
...
loaded ebpf program...
...
```

in another terminal, run the benchmark:

```console
$ LD_PRELOAD=build/runtime/agent/libbpftime-agent.so benchmark/test
attaching prog 3 to fd 4
Successfully attached

a[b] + c for 100000 times
Elapsed time: 0.044516977 seconds
avg function elapse time: 445.169770 ns
```

## userspace uretprobe

run the uretprobe:

```console
$ LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so benchmark/uretprobe/uretprobe
manager constructed
global_shm_open_type 0 for bpftime_maps_shm
Closing 3
libbpf: loading object 'uprobe_bpf' from buffer
libbpf: elf: section(2) .symtab, size 120, link 1, flags 0, type=2
...
loaded ebpf program...
...
```

in another terminal, run the benchmark:

```console
$ LD_PRELOAD=build/runtime/agent/libbpftime-agent.so benchmark/test
attaching prog 3 to fd 4
Successfully attached

a[b] + c for 100000 times
Elapsed time: 0.047297222 seconds
avg function elapse time: 472.972220 ns
```
