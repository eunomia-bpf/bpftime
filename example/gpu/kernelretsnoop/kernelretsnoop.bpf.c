#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP 1502
#define BPF_MAP_TYPE_GPU_RINGBUF_MAP 1527

struct data {
	u64 block_x, block_y, block_z;
	u64 thread_x, thread_y, thread_z;
	u64 timestamp;
};

struct {
	__uint(type, BPF_MAP_TYPE_GPU_RINGBUF_MAP);
	__uint(max_entries, 256);
	__type(key, u32);
	__type(value, struct data);
} rb SEC(".maps");

static const void (*ebpf_puts)(const char *) = (void *)501;
static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int cuda__retprobe()
{
	struct data data = {};

	bpf_get_block_idx(&data.block_x, &data.block_y, &data.block_z);
	bpf_get_thread_idx(&data.thread_x, &data.thread_y, &data.thread_z);
	data.timestamp = bpf_get_globaltimer();
	return bpf_perf_event_output(NULL, &rb, 0, &data,
				     sizeof(struct data));
}

char LICENSE[] SEC("license") = "GPL";
