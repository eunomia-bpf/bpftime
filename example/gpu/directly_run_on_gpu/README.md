# directly_run

A simple example to run eBPF program directly on GPU

```
make -j8
bpftime load ./directly_run &
bpftimetool run-on-cuda cuda__run
```
