#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "cudatest.bpf.h"
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, u32);
	__type(value, u64);
} libc_malloc_calls_total SEC(".maps");

struct map_key_type key_cpu;
struct map_key_type key_gpu;

struct {
	__uint(type, BPF_MAP_TYPE_LRU_HASH);
	__uint(max_entries, 64);
	__type(key, struct map_key_type);
	__type(value, u64);
} lru_map SEC(".maps");

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

	u64 rand = bpf_get_prandom_u32() % 256;
	key_cpu.data[0] = rand;
	u32 value = 1;
	bpf_map_update_elem(&lru_map, &key_cpu, &value, 0);
	return 0;
}

SEC("uprobe/libc.so.6:free")
int do_count__cuda(struct pt_regs *ctx)
{
	// increment_map(&libc_malloc_calls_total, &key, 2);
	// struct map_key_type key;
	// for (int i = 0; i < sizeof(key.data) / sizeof(key.data[0]); i++)
	// 	key.data[i] = 0;
	for (int i = 0; i < 100; i++) {
		key_gpu.data[0] = i;
		bpf_map_lookup_elem(&lru_map, &key_gpu);
	}
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
