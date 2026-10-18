#define BPF_NO_GLOBAL_DATA
#include "vmlinux.h"
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_helpers.h>

SEC("uprobe")
__u64 BPF_KPROBE(my_function_uprobe, int number, char *str, char ch)
{
	// Exercise all three target register offsets, including dereferencing
	// PARM2, without an unsafe formatting/stdio helper in SIGTRAP.
	if (number != 1 || str[0] != 'h' || str[1] != 'e' || ch != 'c')
		return 0;
	return bpf_get_func_ip(ctx);
}

SEC("uretprobe")
__u64 BPF_KRETPROBE(my_function_uretprobe)
{
	return bpf_get_func_ip(ctx);
}

SEC("uprobe")
__u64 BPF_KPROBE(strdup_uprobe, char *str)
{
	return str[0] == 'a' && str[1] == 'a' ? 0 : 1;
}

SEC("uretprobe")
__u64 BPF_KRETPROBE(strdup_uretprobe, char *str)
{
	return str[0] == 'a' && str[1] == 'a' ? 0 : 1;
}

char LICENSE[] SEC("license") = "Dual BSD/GPL";
