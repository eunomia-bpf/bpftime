#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "ufunc.bpf.h"

SEC("uprobe/./victim:target_func")
int do_uprobe_trace(struct pt_regs *ctx)
{
	bpf_printk("target_func called.\n");
	void *ptr = (void*)UFUNC_CALL_NAME_1("ufunc_malloc", 1024);
	UFUNC_CALL_NAME_1("ufunc_free", ptr);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";