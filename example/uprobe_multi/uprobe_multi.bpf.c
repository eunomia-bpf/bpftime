#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

SEC("uprobe.multi/./victim:uprobe_multi_func_*")
int uprobe_multi_test(struct pt_regs *ctx)
{
	bpf_printk("Entry triggered: %d, %d", ctx->di, ctx->si);
	return 0;
}

SEC("uretprobe.multi/./victim:uprobe_multi_func_*")
int uretprobe_multi_test(struct pt_regs *ctx)
{
	bpf_printk("Return triggered: %d", ctx->ax);
	return 0;
}

char _license[] SEC("license") = "GPL";
