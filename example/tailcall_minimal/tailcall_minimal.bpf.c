#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

struct {
	__uint(type, BPF_MAP_TYPE_PROG_ARRAY);
	__uint(max_entries, 1024);
	__type(key, u32);
	__type(value, int);
} prog_array SEC(".maps");

SEC("uprobe/./victim:add_func")
int test_func(struct pt_regs *ctx)
{
	bpf_tail_call(ctx, &prog_array, 0);

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
