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

SEC("uprobe/libc.so.6:free")
int do_count__cuda(struct pt_regs *ctx)
{
	u64 key[4] = {1,0,0,0};
	u64 value[4] = {114514,0,0,0};
	// bpf_map_update_elem(&libc_malloc_calls_total,key,key,0);
	// u64* count = bpf_map_lookup_elem(&libc_malloc_calls_total, key);
	// *count += 1;
	u64  *count = bpf_map_lookup_elem(&libc_malloc_calls_total, key);
	if (!count) {
		value[0]=0;
		bpf_map_update_elem(&libc_malloc_calls_total, key, value, BPF_NOEXIST);
		count = bpf_map_lookup_elem(&libc_malloc_calls_total, key);
		if (!count) {
			return 0;
		}
	}
	value[0] = *count + 1;
	bpf_map_update_elem(&libc_malloc_calls_total, key, value, BPF_EXIST);


	key[0] = 2;
	value[0] = 114514;
	bpf_map_update_elem(&libc_malloc_calls_total,key,value,0);

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
