#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

SEC("uprobe/runtime.casgstatus")
int go_trace_test(struct pt_regs *ctx)
{
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
