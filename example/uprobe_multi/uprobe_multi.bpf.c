#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

SEC("uprobe.multi/./victim:uprobe_multi_func_*")
int uprobe_multi_test(struct pt_regs* ctx){
    bpf_printk("Triggered: %ld", ctx->si);
    return 0;
}


char _license[] SEC("license") = "GPL";
