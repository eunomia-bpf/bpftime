#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define DEFINE_MAP_OPERATIONS(map_name, map_type) \
struct { \
    __uint(type, map_type); \
    __uint(max_entries, 1024); \
    __type(key, u32); \
    __type(value, u64); \
} map_name SEC(".maps"); \
\
SEC("uprobe/benchmark/test:__bench_" #map_name "_update") \
int map_name##_update(struct pt_regs *ctx) \
{ \
    for (int i = 0; i < 1000; i++) { \
        u32 key = i; \
        u64 value = i; \
        bpf_map_update_elem(&map_name, &key, &value, BPF_ANY); \
    } \
    return 0; \
} \
\
SEC("uprobe/benchmark/test:__bench_" #map_name "_delete") \
int map_name##_delete(struct pt_regs *ctx) \
{ \
    for (int i = 0; i < 1000; i++) { \
        u32 key = i; \
        bpf_map_delete_elem(&map_name, &key); \
    } \
    return 0; \
} \
\
SEC("uprobe/benchmark/test:__bench_" #map_name "_lookup") \
int map_name##_lookup(struct pt_regs *ctx) \
{ \
    for (int i = 0; i < 1000; i++) { \
        u32 key = i; \
        bpf_map_lookup_elem(&map_name, &key); \
    } \
    return 0; \
}

// Define operations for an array map
DEFINE_MAP_OPERATIONS(array_map, BPF_MAP_TYPE_ARRAY)

// Define operations for a hash map
DEFINE_MAP_OPERATIONS(hash_map, BPF_MAP_TYPE_HASH)

// Define operations for a per-cpu array map
DEFINE_MAP_OPERATIONS(per_cpu_hash_map, BPF_MAP_TYPE_PERCPU_HASH)

// Define operations for a per-cpu hash map
DEFINE_MAP_OPERATIONS(per_cpu_array_map, BPF_MAP_TYPE_PERCPU_ARRAY)

SEC("uprobe/benchmark/test:__bench_write")
int BPF_UPROBE(__bench_write, char *a, int b, uint64_t c)
{
	char buffer[5] = "text";
    for (int i = 0; i < 1000; i++) {
	    bpf_probe_write_user(a, buffer, sizeof(buffer));
    }
	return b + c;
}

SEC("uprobe/benchmark/test:__bench_read")
int BPF_UPROBE(__bench_read, char *a, int b, uint64_t c)
{
	char buffer[5];
    int res;
    for (int i = 0; i < 1000; i++) {
	    bpf_probe_read_user(buffer, sizeof(buffer), a);
    }
	return b + c + res + buffer[1];
}

SEC("uprobe/benchmark/test:__bench_uprobe")
int BPF_UPROBE(__bench_uprobe, char *a, int b, uint64_t c)
{
	return b + c;
}

SEC("uretprobe/benchmark/test:__bench_uretprobe")
int BPF_URETPROBE(__bench_uretprobe, int ret)
{
	return ret;
}

SEC("uprobe/benchmark/test:__bench_uprobe_uretprobe")
int BPF_UPROBE(__bench_uprobe_uretprobe_1, char *a, int b, uint64_t c)
{
	return b + c;
}

SEC("uretprobe/benchmark/test:__bench_uprobe_uretprobe")
int BPF_URETPROBE(__benchmark_test_function_1_2, int ret)
{
	return ret;
}

char LICENSE[] SEC("license") = "GPL";
