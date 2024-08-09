# bpftime-aot cli

An cli for help to compile eBPF to native ELF.

It can be used to compile eBPF insns to native insns with helpers, maps define, or load native ELF to run.

## Usage

```console
# bpftime-aot help
Usage: /home/yunwei/ebpf-xdp-dpdk/build-bpftime/bpftime/tools/aot/bpftime-aot [--help] [--version] {build,compile,run}

Optional arguments:
  -h, --help     shows help message and exits 
  -v, --version  prints version information and exits 

Subcommands:
  build         Build native ELF(s) from eBPF ELF. Each program in the eBPF ELF will be built into a single native ELF
  compile       Compile the eBPF program loaded in shared memory
  run           Run an native eBPF program
```

## Build ELF from shared mnemory and use it with helpers and maps

load the eBPF programs and maps to shared memory:

```sh
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so example/malloc/malloc
```

The eBPF code here is:

```c
#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, u32);
    __type(value, u64);
} libc_malloc_calls_total SEC(".maps");

static int increment_map(void *map, void *key, u64 increment)
{
    u64 zero = 0, *count = bpf_map_lookup_elem(map, key);
    if (!count) {
        bpf_map_update_elem(map, key, &zero, BPF_NOEXIST);
        count = bpf_map_lookup_elem(map, key);
        if (!count) {
            return 0;
        }
    }
    u64 res = *count + increment;
    bpf_map_update_elem(map, key, &res, BPF_EXIST);

    return *count;
}

SEC("uprobe/libc.so.6:malloc")
int do_count(struct pt_regs *ctx)
{
    u32 pid = bpf_get_current_pid_tgid() >> 32;

    bpf_printk("malloc called from pid %d\n", pid);

    increment_map(&libc_malloc_calls_total, &pid, 1);

    return 0;
}

char LICENSE[] SEC("license") = "GPL";
```

then build the native ELF from shared memory:

```sh
bpftime-aot compile
```

You will get a native ELF file named `do_count.o`.

You can link it with your program and execute it:

```sh
cd bpftime/tools/aot/example
clang -O2 main.c do_count.o -o malloc
```

The drive program is like:

```c
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>

int bpf_main(void* ctx, uint64_t size);

// bpf_printk
uint64_t _bpf_helper_ext_0006(uint64_t fmt, uint64_t fmt_size, ...)
{
    const char *fmt_str = (const char *)fmt;
    va_list args;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#pragma GCC diagnostic ignored "-Wvarargs"
    va_start(args, fmt_str);
    long ret = vprintf(fmt_str, args);
#pragma GCC diagnostic pop
    va_end(args);
    return 0;
}

// bpf_get_current_pid_tgid
uint64_t _bpf_helper_ext_0014(void)
{
    static int tgid = -1;
    static int tid = -1;
    if (tid == -1)
        tid = gettid();
    if (tgid == -1)
        tgid = getpid();
    return ((uint64_t)tgid << 32) | tid;
}

// here we use an var to mock the map.
uint64_t counter_map = 0;

// bpf_map_lookup_elem
void * _bpf_helper_ext_0001(void *map, const void *key)
{
    printf("bpf_map_lookup_elem\n");
    return &counter_map;
}

// bpf_map_update_elem
long _bpf_helper_ext_0002(void *map, const void *key, const void *value, uint64_t flags)
{
    printf("bpf_map_update_elem\n");
    if (value == NULL) {
        printf("value is NULL\n");
        return -1;
    }
    uint64_t* value_ptr = (uint64_t*)value_ptr;
    counter_map = *value_ptr;
    printf("counter_map: %lu\n", counter_map);
    return 0;
}

uint64_t __lddw_helper_map_by_fd(uint32_t id) {
    printf("map_by_fd\n");
    return 0;
}

int main() {
    printf("Hello, World!\n");
    bpf_main(NULL, 0);
    return 0;
}
```

Note by loading eBPF programs with libbpf and LD_PRELOAD, maps, global variables, and helpers are already relocated in shared memory, so you can use them directly in your program. For example, the input of `__lddw_helper_map_by_fd` function would be the actual map id in shared memory.

You can refer to `example/malloc.json` for details about how the maps are relocated.

## Compile from eBPF bytecode ELF

You can also compile the eBPF bytecode ELF to native ELF:

```sh
bpftime-aot build bpftime/example/minimal/.output/uprobe.bpf.o -e
```

In this way, the relocation of maps, global variables, and helpers will not be done. The helpers is still works.

## run native ELF

Given a eBPF code:

```c
#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

SEC("uprobe/./victim:target_func")
int do_uprobe_trace(struct pt_regs *ctx)
{
    bpf_printk("target_func called.\n");
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
```

The native C code after relocation is like:

```c
int _bpf_helper_ext_0006(char* arg0);

int bpf_main(void *ctx)
{
    _bpf_helper_ext_0006("target_func called.\n");
    return 0;
}
```

Compile it with `clang -O3 -c -o do_uprobe_trace.o do_uprobe_trace.c`, and you can load it with AOT runtime.

You can simply run the native ELF:

```console
# bpftime-aot run do_uprobe_trace.o 
[2024-03-24 21:57:53.446] [info] [llvm_jit_context.cpp:81] Initializing llvm
[2024-03-24 21:57:53.446] [info] [llvm_jit_context.cpp:204] LLVM-JIT: Loading aot object
target_func called.
[2024-03-24 21:57:53.449] [info] [main.cpp:190] Output: 0
```

## emit llvm ir

```sh
bpftime-aot compile -e
```

or:

```sh
bpftime-aot build -e minimal.bpf.o
```
