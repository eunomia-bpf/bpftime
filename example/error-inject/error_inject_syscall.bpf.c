#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

SEC("tracepoint/syscalls/sys_enter_openat")
int do_override_inject_syscall_sys_enter_openat(
	struct trace_event_raw_sys_enter *ctx)
{
	int rand = bpf_get_prandom_u32();
	if (rand % 2 == 0) {
		bpf_printk("bpf: Inject error. Target func will not exec.\n");
		bpf_override_return(ctx, -1);
		return 0;
	}
	bpf_printk("bpf: Continue.\n");
	return 0;
}

SEC("tracepoint/syscalls/sys_enter_open")
int do_override_inject_syscall_sys_enter_open(
	struct trace_event_raw_sys_enter *ctx)
{
	int rand = bpf_get_prandom_u32();
	if (rand % 2 == 0) {
		bpf_printk("bpf: Inject error. Target func will not exec.\n");
		bpf_override_return(ctx, -1);
		return 0;
	}
	bpf_printk("bpf: Continue.\n");
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
