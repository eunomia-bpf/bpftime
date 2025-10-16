#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP 1502
#define BPF_MAP_TYPE_GPU_RINGBUF_MAP 1527

struct event_data {
    u64 timestamp;
    u32 type;
    u32 data;
};

// GPU Ring buffer for event collection
struct {
    __uint(type, BPF_MAP_TYPE_GPU_RINGBUF_MAP);
    __uint(max_entries, 16);
    __type(key, u32);
    __type(value, struct event_data);
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

// Per-GPU-thread array map
struct {
    __uint(type, BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u64);
} call_count SEC(".maps");

static const u64 (*bpf_get_globaltimer)(void) = (void *)502;

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
    struct event_data data;
    data.timestamp = 0;
    data.type = 1;
    data.data = 0;
    bpf_perf_event_output(NULL, &events, 0, &data, sizeof(struct event_data));
    return 0;
}

// ============== Global Timer Test ==============

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe_globaltimer()
{
    u64 timer = bpf_get_globaltimer();
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

// ============== Per-GPU-Thread Array Map Tests ==============

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe_pergputd_array_lookup()
{
    u32 key = 0;
    u64 *val = bpf_map_lookup_elem(&call_count, &key);
    if (val) {
        (*val)++;
    }
    return 0;
}

// ============== Memtrace Test ==============

SEC("kprobe/__memcapture")
int cuda__probe_memtrace()
{
    // Basic memory trace - minimal body
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
