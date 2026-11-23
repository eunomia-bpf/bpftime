# Runtime fuzz targets

libFuzzer targets for the bpftime runtime.

## `bpftime_map_ops_fuzzer`

`map_ops_fuzzer.cpp` creates a userspace map in a private shared-memory
segment and runs a sequence of operations on it through `bpf_map_handler`.
That is the dispatch used by the BPF helpers (`from_syscall=false`) and by the
intercepted `bpf()` syscall (`from_syscall=true`).

Input layout:

| Byte(s) | Meaning |
| ------- | ------- |
| 0 | map shape: hash, array, percpu hash, percpu array, LRU hash, queue, stack |
| 1 | key size, `1 + b % 64` (always 4 for arrays, 0 for queue/stack) |
| 2 | value size, `1 + b % 256` |
| 3 | max entries, `1 + b % 64` |
| 4 | creation flags: none, `BPF_F_RDONLY` or `BPF_F_WRONLY` |
| 5.. | operations, up to 256 per input |

Each operation byte holds the access origin (bit 7: syscall), the update/push
flags (bits 4-5: `BPF_ANY`, `BPF_NOEXIST`, `BPF_EXIST`, or the invalid value
3) and the opcode (low bits). It is followed by a key, plus a value when the
operation writes one. Keyed maps support update, lookup, delete,
get-next-key (from a key or from the start) and lookup-and-delete. Queue and
stack maps support push, pop, peek and lookup-and-delete. Bytes past the end
of the input read as zero.

The map attributes stay inside the ranges the kernel accepts at
`BPF_MAP_CREATE`, so a crash points at a map backend and not at a missing
create-time check. The shared-memory segment is named after the process id
and removed at exit.

## Building and running

The targets need clang:

```bash
CC=clang CXX=clang++ cmake -Bbuild-fuzz -DBPFTIME_ENABLE_FUZZING=ON \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-fuzz --target bpftime_map_ops_fuzzer
mkdir -p corpus
./build-fuzz/runtime/fuzz/bpftime_map_ops_fuzzer corpus -max_len=4096
```

`BPFTIME_ENABLE_FUZZING` builds the whole tree with
`-fsanitize=fuzzer-no-link,address,undefined`, so the runtime and map code
get coverage feedback and sanitizer checks. Only the fuzzer executable links
the libFuzzer driver.

The unit-test target `bpftime_runtime_tests` compiles the same entry point
and replays fixed inputs through it (`[fuzz]` tag in
`runtime/unit-test/maps/test_map_ops_fuzzer_replay.cpp`). That keeps the
harness building and running in the regular CI without clang.
