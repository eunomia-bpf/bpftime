#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_NV_GPU_ARRAY_MAP 1502
#define BPF_MAP_TYPE_NV_GPU_RINGBUF_MAP 1527

struct big_struct {
	char s[1024];
};

struct {
	__uint(type, BPF_MAP_TYPE_NV_GPU_RINGBUF_MAP);
	__uint(max_entries, 16);
	__type(key, u32);
	__type(value, struct big_struct);
} rb SEC(".maps");

static const void (*ebpf_puts)(const char *) = (void *)501;
static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

struct data {
	int type;
	union {
		struct {
			u64 x, y, z;
		} thread;
		u64 time;
	};
};

SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int retprobe__cuda()
{
	struct data data;
	u64 begin, end;

	data.type = 0;
	bpf_get_thread_idx(&data.thread.x, &data.thread.y, &data.thread.z);
	begin = bpf_get_globaltimer();
	bpf_perf_event_output(NULL, &rb, 0, &data, sizeof(struct data));
	end = bpf_get_globaltimer();
	data.type = 1;
	data.time = end - begin;
	bpf_perf_event_output(NULL, &rb, 0, &data, sizeof(struct data));
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
