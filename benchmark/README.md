
## Suggest build configuration

```sh
cmake -Bbuild -DLLVM_DIR=/usr/lib/llvm-15/cmake -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -DBPFTIME_LLVM_JIT=1 -DBPFTIME_ENABLE_LTO=1 -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO -DENABLE_PROBE_WRITE_CHECK=0 -DENABLE_PROBE_READ_CHECK=0
cmake --build build --config RelWithDebInfo --target install -j
```

If you fail to build , notice LLVM version.

## build and run at a click

Build the agent first. In project root:

```sh
make build
```

build the benchmark driver:

```sh
make -C benchmark
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
