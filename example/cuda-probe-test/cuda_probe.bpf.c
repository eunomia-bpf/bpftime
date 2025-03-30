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

SEC("kretprobe/test_probe")
int retprobe__cuda(struct pt_regs *ctx)
{

	return 0;
}


SEC("kprobe/test_probe")
int probe__cuda(struct pt_regs *ctx)
{

	return 0;
}


char LICENSE[] SEC("license") = "GPL";
