#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "uprobe_multi.h"

struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 256 * 1024);
} rb SEC(".maps");

SEC("uprobe.multi/./victim:uprobe_multi_func_*")
int uprobe_multi_test(struct pt_regs *ctx)
{
	struct uprobe_multi_event *event;
	event = bpf_ringbuf_reserve(&rb, sizeof(*event), 0);
	if (!event)
		return 0;
	event->is_ret = 0;
	event->uprobe.arg1 = (long)ctx->di;
	event->uprobe.arg2 = (long)ctx->si;
	bpf_ringbuf_submit(event, 0);
	return 0;
}

SEC("uretprobe.multi/./victim:uprobe_multi_func_*")
int uretprobe_multi_test(struct pt_regs *ctx)
{
	struct uprobe_multi_event *event;
	event = bpf_ringbuf_reserve(&rb, sizeof(*event), 0);
	if (!event)
		return 0;
	event->is_ret = 1;
	event->uretprobe.ret_val = (long)ctx->ax;
	bpf_ringbuf_submit(event, 0);
	return 0;
}

char _license[] SEC("license") = "GPL";
