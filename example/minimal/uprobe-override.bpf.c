#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

SEC("uprobe")
int do_uprobe_override_patch(struct pt_regs *ctx)
{
	bpf_printk("target_func called is overrided.\n");
	bpf_override_return(ctx, 0);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
