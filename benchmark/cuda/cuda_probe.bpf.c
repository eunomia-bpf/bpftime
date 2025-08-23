#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int probe__cuda()
{
    // empty probe
    // bpf_printk("Entered _Z9vectorAddPKfS0_Pf\n");
    return 0;
}

SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int retprobe__cuda()
{
    // empty probe
    // bpf_printk("Exited _Z9vectorAddPKfS0_Pf\n");
    return 0;
}

char LICENSE[] SEC("license") = "GPL"; 