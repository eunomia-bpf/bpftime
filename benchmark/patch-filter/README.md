# bench for patch and filter

release:

```
make release
```

## test for uprobe kernel and uprobe userspace

About 10x faster without userspace jit:

- uprobe: 2000ns
- userspace uprobe vm: 200ns
- userspace uprobe ubpf jit: 200ns

```
yunwei@yunwei-virtual-machine:~/bpftime$ sudo build/tools/cli/bpftime-cli benchmark/patch-filter/bench.filter.json --bench-kernel
Unknown ffi_funcs is not array
base line: a[b] + c for 10009 times
Elapsed time: 0.000023963 seconds
base line avg hooked time: 2.394145 ns

after uprobe kernel: a[b] + c for 10009 times
Elapsed time: 0.022514019 seconds
total kernel uprobe time: 0.022490
avg uprobe time: 2246.983315 ns

yunwei@yunwei-virtual-machine:~/bpftime$ sudo build/tools/cli/bpftime-cli benchmark/patch-filter/bench.filter.json --benchmark
Unknown ffi_funcs is not array
base line: a[b] + c for 10009 times
Elapsed time: 0.000040201 seconds
base line avg hooked time: 4.016485 ns

find and load program: __benchmark_test_function
load insn cnt: 5
find function __benchmark_test_function at 0x55acabd78010
attach replace 0x55acabd78010
after uprobe userspace: a[b] + c for 10009 times
Elapsed time: 0.002758535 seconds
total userspace uprobe time: 0.002718
avg userspace uprobe time: 271.588970 ns
```

with ubpf jit:

```
$ sudo build/tools/cli/bpftime-cli benchmark/patch-filter/bench.filter.json --benchmark
Unknown ffi_funcs is not array
base line: a[b] + c for 10009 times
Elapsed time: 0.000023918 seconds
base line avg hooked time: 2.389649 ns

find and load program: __benchmark_test_function
load insn cnt: 5
find function __benchmark_test_function at 0x55f8e04a5010
attach replace 0x55f8e04a5010
after uprobe userspace: a[b] + c for 10009 times
Elapsed time: 0.002357156 seconds
total userspace uprobe time: 0.002333
avg userspace uprobe time: 233.113997 ns

yunwei@yunwei-virtual-machine:~/bpftime$ sudo build/tools/cli/bpftime-cli benchmark/patch-filter/bench.filter.json --benchmark
Unknown ffi_funcs is not array
base line: a[b] + c for 10009 times
Elapsed time: 0.000024025 seconds
base line avg hooked time: 2.400340 ns

find and load program: __benchmark_test_function
load insn cnt: 5
find function __benchmark_test_function at 0x556692534010
attach replace 0x556692534010
after uprobe userspace: a[b] + c for 10009 times
Elapsed time: 0.002379450 seconds
total userspace uprobe time: 0.002355
avg userspace uprobe time: 235.330702 ns
```

## test for filter and patch

- without userspace jit

```
yunwei@yunwei-virtual-machine:~/bpftime$ build/tools/cli/bpftime-cli benchmark/patch-filter/bench.filter.json --benchmark
Unknown ffi_funcs is not array
base line: a[b] + c for 10009 times
Elapsed time: 0.000023924 seconds
find and load program: __benchmark_test_function
load insn cnt: 5
find function __benchmark_test_function at 0x563355abeef5
attach replace 0x563355abeef5
after hooked: a[b] + c for 10009 times
Elapsed time: 0.002532293 seconds
total hooked time: 0.002508
avg hooked time: 250.611350 ns

yunwei@yunwei-virtual-machine:~/bpftime$ build/tools/cli/bpftime-cli benchmark/patch-filter/bench.patch.json --benchmark
Unknown ffi_funcs is not array
base line: a[b] + c for 10009 times
Elapsed time: 0.000023981 seconds
find and load program: __benchmark_test_function_probe
load insn cnt: 26
find function __benchmark_test_function at 0x556f35343ef5
[+] install patch to __benchmark_test_function: 0x556f35343ef5
attach replace 0x556f35343ef5
after hooked: a[b] + c for 10009 times
Elapsed time: 0.004181585 seconds
total hooked time: 0.004158
avg hooked time: 415.386552 ns
yunwei@yunwei-virtual-machine:~/bpftime$
```
