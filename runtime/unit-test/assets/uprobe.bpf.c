#define BPF_NO_GLOBAL_DATA
#include "vmlinux.h"
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_helpers.h>

#ifndef BPF_UPROBE
#define BPF_UPROBE BPF_KPROBE
#endif
#ifndef BPF_URETPROBE
#define BPF_URETPROBE BPF_KRETPROBE
#endif

SEC("uprobe")
int BPF_UPROBE(my_function_uprobe, int parm1, char* str, char c) {
	bpf_printk("BPF_UPROBE: %d %s %c\n", parm1, str, c);
	return 0;
}

SEC("uprobe")
int BPF_URETPROBE(my_function_uretprobe) {
	bpf_printk("BPF_URETPROBE\n");
	return 0;
}

SEC("uprobe")
int BPF_UPROBE(strdup_uprobe, char* str) {
	bpf_printk("Hello from strdup: %s\n", str);
	return 0;
}

SEC("uretprobe")
int BPF_URETPROBE(strdup_uretprobe, char* str) {
	bpf_printk("Hello from strdup ret: %s\n", str);
	return 0;
}

char LICENSE[] SEC("license") = "Dual BSD/GPL";

