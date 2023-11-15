# benchmark of hash maps

- __benchmark_test_function1: hashmap bpf_map_lookup_elem
- __benchmark_test_function2: hashmap bpf_map_delete_elem
- __benchmark_test_function3: hashmap bpf_map_update_elem

run the uprobe:

```console
$ LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so benchmark/hash_map/uprobe
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

Benchmarking __benchmark_test_function1
a[b] + c for 100000 times
Elapsed time: 0.038217773 seconds
Average time usage 382.177730 ns

Benchmarking __benchmark_test_function2
a[b] + c for 100000 times
Elapsed time: 0.020004455 seconds
Average time usage 200.044550 ns

Benchmarking __benchmark_test_function3
a[b] + c for 100000 times
Elapsed time: 0.047916014 seconds
Average time usage 479.160140 ns

INFO [34534]: Global shm destructed
```
