# minimal examples

- `uprobe/uretprobe`: trace userspace functions at start or and. No affect the control flow.
- `ureplace`: replace the userspace function with a eBPF function
- `ufilter`: filter the userspace function
- `syscall tracepoints`: trace the specific syscall types. No affect the control flow of syscalls.

## uprobe trace

This code is a BPF (Berkeley Packet Filter) program written in C, often used for tracing and monitoring activities in the Linux kernel. BPF allows you to run custom programs within the kernel without modifying its source code. The code you provided creates a BPF program that uses a BPF map to count the number of times the `uprobe` function is called within a specified cgroup.

```c
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include "bits.bpf.h"
#include "maps.bpf.h"

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, u64);
    __type(value, u64);
} libc_uprobe_calls_total SEC(".maps");

SEC("uprobe/libc.so.6:uprobe")
int do_count(struct pt_regs *ctx)
{
    u64 cgroup_id = bpf_get_current_cgroup_id();

    increment_map(&libc_uprobe_calls_total, &cgroup_id, 1);

    return 0;
}

char LICENSE[] SEC("license") = "GPL";
```

from <https://github.com/cloudflare/ebpf_exporter/blob/master/examples/uprobe.bpf.c>

Here's a breakdown of the code:

1. **Headers Inclusion**:
   - `<vmlinux.h>`: Provides access to kernel data structures and definitions.
   - `<bpf/bpf_helpers.h>`: Includes helper functions and macros for BPF programs.
   - `"bits.bpf.h"`: Custom header file (assumed to contain additional definitions).
   - `"maps.bpf.h"`: Custom header file (assumed to contain definitions related to BPF maps).

2. **Definition of BPF Map**:
   The code defines a BPF map named `libc_uprobe_calls_total` using the `struct` syntax. This map is of type `BPF_MAP_TYPE_HASH` (hash map) with a maximum of 1024 entries. The keys and values are of type `u64` (unsigned 64-bit integer).

3. **Map Definition Attributes**:
   The attributes specified within the map definition (`__uint`, `__type`) set properties of the map, such as its type, maximum number of entries, and types of keys and values.

4. **BPF Program**:
   - The program is associated with a `uprobe` on the `uprobe` function in the `libc.so.6` library.
   - The `do_count` function is executed when the `uprobe` function is called.
   - It retrieves the current cgroup ID using `bpf_get_current_cgroup_id()`.
   - Then, it increments the `libc_uprobe_calls_total` map with the cgroup ID as the key and increments the associated value by 1.

5. **License Information**:
   The `LICENSE[]` array contains the license information for the BPF program. In this case, the program is licensed under the GPL (GNU General Public License).

The purpose of this BPF program is to track and count the number of `uprobe` calls that occur within specific cgroups in the Linux kernel. It uses a BPF hash map to store and update the counts. This can be useful for monitoring memory allocation patterns and resource usage within different cgroups.

### how to run uprobe

server

```sh
LD_PRELOAD=~/.bpftime/libbpftime-syscall-server.so example/minimal/uprobe
```

client

```sh
LD_PRELOAD=~/.bpftime/libbpftime-agent.so example/minimal/victim
```

## Syscall

### how to run syscall

server

```sh
LD_PRELOAD=~/.bpftime/libbpftime-syscall-server.so ./uprobe
```

client

```sh
LD_PRELOAD=~/.bpftime/libbpftime-agent.so ./victim
```

## ureplace

Run server:

```sh
SPDLOG_LEVEL=Debug LD_PRELOAD=~/.bpftime/libbpftime-syscall-server.so ./ureplace
```

Run victim:

```sh
SPDLOG_LEVEL=Debug LD_PRELOAD=~/.bpftime/libbpftime-syscall-server.so ./ureplace
```