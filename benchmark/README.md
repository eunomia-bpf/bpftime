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

## Suggest build configuration

```sh
cmake -Bbuild -DLLVM_DIR=/usr/lib/llvm-15/cmake -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -DBPFTIME_LLVM_JIT=1 -DBPFTIME_ENABLE_LTO=1
cmake --build build --config RelWithDebInfo --target install -j
```

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

```sh
benchmark/test
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

```sh
sudo benchmark/uprobe/uprobe
```

in another terminal, run the benchmark:

```sh
benchmark/test
```

The uprobe or uretprobe function we used is like:

```c
SEC("uprobe/benchmark/test:__benchmark_test_function3")
int BPF_UPROBE(__benchmark_test_function, const char *a, int b, uint64_t c)
{
 return b + c;
}
```

## userspace uprobe

run the uprobe:

```sh
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so benchmark/uprobe/uprobe
```

in another terminal, run the benchmark:

```sh
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so benchmark/test
```

If errors like:

```txt
terminate called after throwing an instance of 'boost::interprocess::interprocess_exception'
  what():  File exists
Aborted (core dumped)
```

happpens, try to use `sudo` mode.

## embed runtime

```sh
build/benchmark/simple-benchmark-with-embed-ebpf-calling
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

## Test syscall trace and untrace

run the test:

```sh
bash ./benchmark/syscount/test.sh
```

result:

```txt
# baseline, no trace syscall
Average read() time over 10 runs: 349 ns
Average sendmsg() time over 10 runs: 3640 ns
# trace with syscount
Average read() time over 10 runs: 437 ns
Average sendmsg() time over 10 runs: 3952 ns
# filter out the pid
Average read() time over 10 runs: 398 ns
Average sendmsg() time over 10 runs: 3690 ns
# trace with userspace syscall tracepoint
Average read() time over 10 runs: 531 ns
Average sendmsg() time over 10 runs: 3681 ns
```

## Results on another machine

kernel:

```txt
Benchmarking __bench_uprobe_uretprobe in thread 1
Average time usage 3060.196770 ns, iter 100000 times

Benchmarking __bench_uretprobe in thread 1
Average time usage 2958.493390 ns, iter 100000 times

Benchmarking __bench_uprobe in thread 1
Average time usage 1910.731360 ns, iter 100000 times

Benchmarking __bench_read in thread 1
Average time usage 1957.552190 ns, iter 100000 times

Benchmarking __bench_write in thread 1
Average time usage 1955.735460 ns, iter 100000 times
```

Userspace:

```txt
Benchmarking __bench_uprobe_uretprobe in thread 1
Average time usage 391.967450 ns, iter 100000 times

Benchmarking __bench_uretprobe in thread 1
Average time usage 383.851670 ns, iter 100000 times

Benchmarking __bench_uprobe in thread 1
Average time usage 380.935190 ns, iter 100000 times

Benchmarking __bench_read in thread 1
Average time usage 383.135720 ns, iter 100000 times

Benchmarking __bench_write in thread 1
Average time usage 389.037170 ns, iter 100000 times
```
