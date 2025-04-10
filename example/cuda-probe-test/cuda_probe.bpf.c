#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>


struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, u32);
	__type(value, u64);
} test_hash_map SEC(".maps");

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

SEC("kretprobe/vprintf")
int retprobe__cuda(struct pt_regs *ctx)
{
	u64 key = 12345;
	increment_map(&test_hash_map, &key, 1);
	return 0;
}


SEC("kprobe/vprintf")
int probe__cuda(struct pt_regs *ctx)
{
	u64 key = 12345;
	increment_map(&test_hash_map, &key, 1);
	return 0;
}


char LICENSE[] SEC("license") = "GPL";
