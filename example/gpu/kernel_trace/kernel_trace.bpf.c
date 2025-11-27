#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_RINGBUF_MAP 1527

struct kernel_trace_event {
	u64 block_x, block_y, block_z;
	u64 thread_x, thread_y, thread_z;
	u64 globaltimer;
};

struct {
	__uint(type, BPF_MAP_TYPE_GPU_RINGBUF_MAP);
	__uint(max_entries, 128);
	__type(key, u32);
	__type(value, struct kernel_trace_event);
} events SEC(".maps");

static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__kernel_trace()
{
	struct kernel_trace_event ev = {};

	bpf_get_block_idx(&ev.block_x, &ev.block_y, &ev.block_z);
	bpf_get_thread_idx(&ev.thread_x, &ev.thread_y, &ev.thread_z);
	ev.globaltimer = bpf_get_globaltimer();
	bpf_perf_event_output(NULL, &events, 0, &ev, sizeof(ev));
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
