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

SEC("uprobe/benchmark/test:__benchmark_test_function3")
int test_update(struct pt_regs *ctx)
{
	u32 key = 0;
	u64 value = 0;
	bpf_map_update_elem(&libc_malloc_calls_total, &key, &value, 0);

	return 0;
}

SEC("uprobe/benchmark/test:__benchmark_test_function2")
int test_delete(struct pt_regs *ctx)
{
	u32 key = 0;
	u64 value = 0;
	bpf_map_delete_elem(&libc_malloc_calls_total, &key);

	return 0;
}

SEC("uprobe/benchmark/test:__benchmark_test_function1")
int test_lookup(struct pt_regs *ctx)
{
	u32 key = 0;
	u64 value = 0;
	bpf_map_lookup_elem(&libc_malloc_calls_total, &key);

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
