
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/usdt.bpf.h>

SEC("usdt")
int BPF_USDT(simple_probe, int x, int y, int z)
{
	bpf_printk("bpf: %d + %d = %d\n", x, y, z);
	return 0;
}

char LICENSE[] SEC("license") = "Dual BSD/GPL";
