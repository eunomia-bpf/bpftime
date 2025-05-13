#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "cuda_neuro_surgeon.bpf.h"

// BPF map to store CUDA memory allocation statistics
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, u32);
    __type(value, u64);
} libc_malloc_calls_total SEC(".maps");

// BPF program to track CUDA memory allocations
SEC("uprobe/libc.so.6/malloc")
int do_count(struct pt_regs *ctx)
{
    u32 key = 0;
    u64 *valp, val = 1;

    valp = bpf_map_lookup_elem(&libc_malloc_calls_total, &key);
    if (valp) {
        __u64 new_val = *valp + val;
        bpf_map_update_elem(&libc_malloc_calls_total, &key, &new_val, BPF_ANY);
    } else {
        bpf_map_update_elem(&libc_malloc_calls_total, &key, &val, BPF_ANY);
    }

    return 0;
}

// BPF program to track CUDA-specific memory allocations
SEC("uprobe/libcuda.so/cudaMalloc")
int do_count__cuda(struct pt_regs *ctx)
{
    u32 key = 1;  // Different key for CUDA allocations
    u64 *valp, val = 1;

    valp = bpf_map_lookup_elem(&libc_malloc_calls_total, &key);
    if (valp) {
        __u64 new_val = *valp + val;
        bpf_map_update_elem(&libc_malloc_calls_total, &key, &new_val, BPF_ANY);
    } else {
        bpf_map_update_elem(&libc_malloc_calls_total, &key, &val, BPF_ANY);
    }

    return 0;
}

char LICENSE[] SEC("license") = "GPL";
