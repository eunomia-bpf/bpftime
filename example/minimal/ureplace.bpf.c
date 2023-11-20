#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

SEC("ureplace/example/minimal/victim:target_func")
int do_ureplace_patch(struct pt_regs *ctx)
{
	bpf_printk("target_func called for replaced.\n");
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
