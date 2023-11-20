#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

enum filter_op {
	OP_SKIP,
	OP_RESUME,
};

SEC("uprobe")
int do_ufilter_patch(struct pt_regs *ctx)
{
	bpf_printk("target_func called for filtered.\n");
	return 1;
}

char LICENSE[] SEC("license") = "GPL";
