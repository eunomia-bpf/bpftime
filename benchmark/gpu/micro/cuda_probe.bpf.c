#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

// Ring buffer for event collection
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

// Array map for storing counters
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1024);
    __type(key, u32);
    __type(value, u64);
} array_map SEC(".maps");

// Hash map for key-value storage
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, u32);
    __type(value, u64);
} hash_map SEC(".maps");

// Per-CPU array map
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1024);
    __type(key, u32);
    __type(value, u64);
} per_cpu_array_map SEC(".maps");

// Per-CPU hash map
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_HASH);
    __uint(max_entries, 1024);
    __type(key, u32);
    __type(value, u64);
} per_cpu_hash_map SEC(".maps");

struct event {
    u32 type;
    u32 data;
};

// ============== Empty Probes (Baseline) ==============

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe()
{
    // Empty probe - minimal overhead baseline
    return 0;
}

SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int cuda__retprobe()
{
    // Empty probe - minimal overhead baseline
    return 0;
}

// ============== Entry-Only Probes ==============

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe_entry()
{
    u32 key = 0;
    u64 val = 1;
    return 0;
}

// ============== Exit-Only Probes ==============

SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int cuda__retprobe_exit()
{
    u32 key = 1;
    u64 val = 1;
    return 0;
}

// ============== Entry + Exit Probes ==============

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe_both_entry()
{
    u32 key = 0;
    u64 val = 1;
    return 0;
}

SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int cuda__retprobe_both_exit()
{
    u32 key = 1;
    u64 val = 1;
    return 0;
}

// ============== Ring Buffer Tests ==============

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe_ringbuf()
{
    struct event *e;
    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (e) {
        e->type = 1;
        e->data = 0;
        bpf_ringbuf_submit(e, 0);
    }
    return 0;
}

// ============== Array Map Tests ==============

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe_array_update()
{
    u32 key = 0;
    u64 val = 1;
    bpf_map_update_elem(&array_map, &key, &val, BPF_ANY);
    return 0;
}

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe_array_lookup()
{
    u32 key = 0;
    u64 *val = bpf_map_lookup_elem(&array_map, &key);
    if (val) {
        (*val)++;
    }
    return 0;
}

// ============== Hash Map Tests ==============

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe_hash_update()
{
    u32 key = 0;
    u64 val = 1;
    bpf_map_update_elem(&hash_map, &key, &val, BPF_ANY);
    return 0;
}

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe_hash_lookup()
{
    u32 key = 0;
    u64 *val = bpf_map_lookup_elem(&hash_map, &key);
    if (val) {
        (*val)++;
    }
    return 0;
}

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe_hash_delete()
{
    u32 key = 0;
    bpf_map_delete_elem(&hash_map, &key);
    return 0;
}

// ============== Per-CPU Array Map Tests ==============

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe_percpu_array_update()
{
    u32 key = 0;
    u64 val = 1;
    bpf_map_update_elem(&per_cpu_array_map, &key, &val, BPF_ANY);
    return 0;
}

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe_percpu_array_lookup()
{
    u32 key = 0;
    u64 *val = bpf_map_lookup_elem(&per_cpu_array_map, &key);
    if (val) {
        (*val)++;
    }
    return 0;
}

// ============== Per-CPU Hash Map Tests ==============

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe_percpu_hash_update()
{
    u32 key = 0;
    u64 val = 1;
    bpf_map_update_elem(&per_cpu_hash_map, &key, &val, BPF_ANY);
    return 0;
}

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe_percpu_hash_lookup()
{
    u32 key = 0;
    u64 *val = bpf_map_lookup_elem(&per_cpu_hash_map, &key);
    if (val) {
        (*val)++;
    }
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
