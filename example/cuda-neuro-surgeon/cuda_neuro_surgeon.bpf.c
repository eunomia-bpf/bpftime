#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "cuda_neuro_surgeon.bpf.h"

// ---------------------------------------------------------------------------
// LRU Tensor Cache Implementation
// ---------------------------------------------------------------------------

// Entry for each tensor in the LRU cache
struct tensor_entry {
    u64 addr;
    u64 size;
};
struct trace_event_raw_cuda_tensor_access {
    unsigned long long tensor_id;  // u64
    unsigned long long addr;       // u64
    unsigned long long size;       // u64
};
// Define an LRU hash map for automatic eviction of least-recently-used tensors
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 256);
    __type(key, u64);              // tensor ID
    __type(value, struct tensor_entry);
} lru_tensor_cache SEC(".maps");

// ---------------------------------------------------------------------------
// LFU Tensor Cache Implementation
// ---------------------------------------------------------------------------

// Statistics for each tensor in the LFU cache
struct tensor_stats {
    u64 addr;
    u64 size;
    u64 freq;     // access frequency counter
};

// Define a hash map to hold tensor stats (frequency counts)
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 256);
    __type(key, u64);              // tensor ID
    __type(value, struct tensor_stats);
} lfu_tensor_cache SEC(".maps");

// ---------------------------------------------------------------------------
// Example Probe: Update both LRU and LFU on tensor load events
// ---------------------------------------------------------------------------

// Hypothetical tracepoint for tensor accesses; adjust name/signature as needed
SEC("kretprobe/tensor_access")
int handle_tensor_access(struct trace_event_raw_cuda_tensor_access *ctx) {
    u64 tensor_id = ctx->tensor_id;
    struct tensor_entry lru_entry = {
        .addr = ctx->addr,
        .size = ctx->size,
    };

    // 1) Update LRU cache: auto-evicts oldest when full
    bpf_map_update_elem(&lru_tensor_cache, &tensor_id, &lru_entry, BPF_ANY);

    // 2) Update LFU cache: increment frequency or insert new
    struct tensor_stats *stats = bpf_map_lookup_elem(&lfu_tensor_cache, &tensor_id);
    if (stats) {
        // Atomically increment frequency counter
        __sync_fetch_and_add(&stats->freq, 1);
    } else {
        struct tensor_stats new_stats = {
            .addr = ctx->addr,
            .size = ctx->size,
            .freq = 1,
        };
        bpf_map_update_elem(&lfu_tensor_cache, &tensor_id, &new_stats, BPF_ANY);
    }

    return 0;
}

char LICENSE[] SEC("license") = "GPL";
