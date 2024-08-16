# benchmark of uprobe and uretprobe

With userspace eBPF runntime, we can:

- Speed up the uprobe and uretprobe by approximate `10x`
- The userspace read and write user memory is approximate `10x` faster than kernel (~5ns vs ~50ns)
- With out any kernel patch or modify the tracing eBPF program
- No privilege is needed for running the eBPF tracing program.

Probes:

| Probe/Tracepoint Types | Kernel (ns)  | Userspace (ns) | Insn Count |
|------------------------|-------------:|---------------:|---------------:|
| Uprobe                 | 3224.172760  | 314.569110     | 4    |
| Uretprobe              | 3996.799580  | 381.270270     | 2    |
| Syscall Tracepoint     | 151.82801    | 232.57691      | 4    |
| Embedding runtime      | Not avaliable |  110.008430   | 4    |

Read and write user memory:

| Probe/Tracepoint Types  | Kernel (ns)     | Userspace (ns) |
|-------------------------|----------------:|---------------:|
| bpf_probe_read - uprobe  | 46.820830       | 2.200530       |
| bpf_probe_write_user - uprobe | 45.004100  | 8.101980       |

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

## Test syscall trace and untrace with syscount

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

## Results for uprobe, uretprobe, and syscall tracepoint

| Probe/Tracepoint Types | Kernel (ns)  | Userspace (ns) | Insn Count |
|------------------------|-------------:|---------------:|---------------:|
| Uprobe                 | 3224.172760  | 314.569110     | 4    |
| Uretprobe              | 3996.799580  | 381.270270     | 2    |
| Syscall Tracepoint     | 151.82801    | 232.57691      | 4    |
| Embedding runtime      | Not avaliable |  110.008430   | 4    |

Tested on `6.2.0-32-generic` kernel and `Intel(R) Core(TM) i7-11800H CPU @ 2.30GHz`.

## Results on another machine

Tested on `kernel version 6.2` and `Intel(R) Xeon(R) Gold 5418Y` CPU.

### Uprobe and read/write with `bpf_probe_write_user` and `bpf_probe_read_user`

Userspace:

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

### maps operations

Run the map op 1000 times in one function. Userspace map op is also faster than the kernel in the current version. Current version is 10x faster than stupid old version.

```c
SEC("uprobe/benchmark/test:__bench_hash_map_lookup")
int test_lookup(struct pt_regs *ctx)
{
    for (int i = 0; i < 1000; i++) {
        u32 key = i;
        u64 value = i;
        bpf_map_lookup_elem(&test_hash_map, &key);
    }
    return 0;
}
```

Kernel map op cost:

```txt

Benchmarking __bench_hash_map_update in thread 1
Average time usage 64738.264680 ns, iter 100000 times

Benchmarking __bench_hash_map_lookup in thread 1
Average time usage 17805.898280 ns, iter 100000 times

Benchmarking __bench_hash_map_delete in thread 1
Average time usage 21795.665340 ns, iter 100000 times

Benchmarking __bench_array_map_update in thread 1
Average time usage 11449.295960 ns, iter 100000 times

Benchmarking __bench_array_map_lookup in thread 1
Average time usage 2093.886500 ns, iter 100000 times

Benchmarking __bench_array_map_delete in thread 1
Average time usage 2126.820310 ns, iter 100000 times

Benchmarking __bench_per_cpu_hash_map_update in thread 1
Average time usage 35050.915650 ns, iter 100000 times

Benchmarking __bench_per_cpu_hash_map_lookup in thread 1
Average time usage 15999.969590 ns, iter 100000 times

Benchmarking __bench_per_cpu_hash_map_delete in thread 1
Average time usage 21664.294940 ns, iter 100000 times

Benchmarking __bench_per_cpu_array_map_update in thread 1
Average time usage 10886.969860 ns, iter 100000 times

Benchmarking __bench_per_cpu_array_map_lookup in thread 1
Average time usage 2749.468760 ns, iter 100000 times

Benchmarking __bench_per_cpu_array_map_delete in thread 1
Average time usage 2778.679460 ns, iter 100000 times
```

Userspace map op cost:

```txt
Benchmarking __bench_hash_map_update in thread 1
Average time usage 30676.986820 ns, iter 100000 times

Benchmarking __bench_hash_map_lookup in thread 1
Average time usage 23486.304570 ns, iter 100000 times

Benchmarking __bench_hash_map_delete in thread 1
Average time usage 13435.901160 ns, iter 100000 times

Benchmarking __bench_array_map_update in thread 1
Average time usage 7081.922160 ns, iter 100000 times

Benchmarking __bench_array_map_lookup in thread 1
Average time usage 4685.450360 ns, iter 100000 times

Benchmarking __bench_array_map_delete in thread 1
Average time usage 6367.443010 ns, iter 100000 times

Benchmarking __bench_per_cpu_hash_map_update in thread 1
Average time usage 95918.602090 ns, iter 100000 times

Benchmarking __bench_per_cpu_hash_map_lookup in thread 1
Average time usage 63294.791110 ns, iter 100000 times

Benchmarking __bench_per_cpu_hash_map_delete in thread 1
Average time usage 460207.364100 ns, iter 100000 times

Benchmarking __bench_per_cpu_array_map_update in thread 1
Average time usage 26109.863360 ns, iter 100000 times

Benchmarking __bench_per_cpu_array_map_lookup in thread 1
Average time usage 9139.355980 ns, iter 100000 times

Benchmarking __bench_per_cpu_array_map_delete in thread 1
Average time usage 5203.339320 ns, iter 100000 times
```

The benchmark without inline the map op function:

| Map Operation                      | Kernel (op - uprobe) (ns) | Userspace (op - uprobe) (ns) |
|------------------------------------|--------------------------:|-----------------------------:|
| __bench_hash_map_update            | 62827.533320              | 30296.051630                 |
| __bench_hash_map_lookup            | 15895.166920              | 23005.369380                 |
| __bench_hash_map_delete            | 19884.933980              | 13054.965970                 |
| __bench_array_map_update           | 9538.564600               | 6701.987970                  |
| __bench_array_map_lookup           |  183.155140               | 4305.515170                  |
| __bench_array_map_delete           |  216.088950               | 5987.507820                  |
| __bench_per_cpu_hash_map_update    | 33140.184290              | 95537.666900                 |
| __bench_per_cpu_hash_map_lookup    | 14089.238230              | 62913.855920                 |
| __bench_per_cpu_hash_map_delete    | 19753.563580              | 459826.428910                |
| __bench_per_cpu_array_map_update   |  8885.238500              | 25728.928170                 |
| __bench_per_cpu_array_map_lookup   |  1838.737400              | 8759.420790                  |
| __bench_per_cpu_array_map_delete   |  1867.948100              | 4802.404130                  |

- Some overhead can be reduced by inlining the map op function.
- We need to fix the performance issue of the per-cpu map in the userspace runtime.