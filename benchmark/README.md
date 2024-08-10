# benchmark of uprobe and uretprobe

With userspace eBPF runntime, we can:

- speed up the uprobe and uretprobe by approximate 10x
- with out any kernel patch or modify the tracing eBPF program
- No privilege is needed for running the eBPF tracing program.

| Probe/Tracepoint Types | Kernel (ns)  | Userspace (ns) | Insn Count |
|------------------------|-------------:|---------------:|---------------:|
| Uprobe                 | 3224.172760  | 314.569110     | 4    |
| Uretprobe              | 3996.799580  | 381.270270     | 2    |
| Syscall Tracepoint     | 151.82801    | 232.57691      | 4    |
| Embedding runtime      | Not avaliable |  110.008430   | 4    |

## build and run at a click

Build the agent first. In project root:

```sh
make build
```

build the benchmark driver:

```sh
make -C benchmark
```

Run the uprobe bench:

```sh
cd benchmark
python3 run_benchmark.py
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
Elapsed time: 0.322417276 seconds
avg function elapse time: 3224.172760 ns
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
avg function elapse time: 3996.799580 ns
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
Elapsed time: 0.031456911 seconds
avg function elapse time: 314.569110 ns
```

If errors like:

```txt
terminate called after throwing an instance of 'boost::interprocess::interprocess_exception'
  what():  File exists
Aborted (core dumped)
```

happpens, try to use `sudo` mode.

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
Elapsed time: 0.038127027 seconds
avg function elapse time: 381.270270 ns
```

## embed runtime

```console
$ build/benchmark/simple-benchmark-with-embed-ebpf-calling
uprobe elf: /home/yunwei/bpftime/build/benchmark/uprobe_prog.bpf.o
uretprobe elf:/home/yunwei/bpftime/build/benchmark/uretprobe_prog.bpf.o
a[b] + c for 100000 times
Elapsed time: 0.011000843 seconds
avg function elapse time: 110.008430 ns
```

## userspace syscall

### run

```sh
sudo ~/.bpftime/bpftime load benchmark/syscall/syscall
```

in another shell, run the target program with eBPF inside:

```sh
sudo ~/.bpftime/bpftime start -s benchmark/syscall/victim
```

- baseline: Average time usage 938.53511ns,  count 1000000
- userspace syscall tracepoint: Average time usage 1489.04251ns,  count 1000000
- kernel tracepoint：Average time usage 1499.47708ns,  count 1000000

You can use python script to run the benchmark:

```console
python3 benchmark/tools/driving.py
```

## Benchmark runner

### Usage
- `make -C ./benchmark`
- `python3 ./benchmark/run_benchmark.py`
