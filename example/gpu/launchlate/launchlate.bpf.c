#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503
#define BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP 1513

static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

#define HIST_BINS 10
#define LAUNCH_QUEUE_SIZE 4096

// Histogram of launch-to-entry latencies. Written on the device, read by the
// host loader.
struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, HIST_BINS);
	__type(key, u32);
	__type(value, u64);
} time_histogram SEC(".maps");

// FIFO of host launch timestamps. The host uprobe enqueues a calibrated
// timestamp; the device probe consumes it in launch order. Shared host<->GPU
// buffer.
struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP);
	__uint(max_entries, LAUNCH_QUEUE_SIZE);
	__type(key, u32);
	__type(value, u64);
} launch_times SEC(".maps");

// queue_state[0] = host write sequence, [1] = device read sequence,
// [2] = underflows, [3] = overflows. Shared host<->GPU counter state.
struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP);
	__uint(max_entries, 4);
	__type(key, u32);
	__type(value, u64);
} queue_state SEC(".maps");

// clock_offset[0] = CLOCK_REALTIME - CLOCK_MONOTONIC, written by the loader and
// read by the host uprobe to place host timestamps in the wall-clock domain.
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 32);
	__type(key, u32);
	__type(value, s64);
} clock_offset SEC(".maps");

// The userspace loader attaches this uprobe to the exact host launch stub for the
// selected CUDA kernel, not to the generic cuLaunchKernel API.
SEC("uprobe")
int BPF_KPROBE(uprobe_cuda_launch)
{
	u64 ts_mono = bpf_ktime_get_ns();
	u32 key = 0;
	s64 *offset_ptr;
	u64 ts_calibrated = ts_mono;

	offset_ptr = bpf_map_lookup_elem(&clock_offset, &key);
	if (offset_ptr)
		ts_calibrated = ts_mono + (u64)*offset_ptr;

	u64 *write_seq = bpf_map_lookup_elem(&queue_state, &key);
	if (!write_seq)
		return 0;
	u64 seq = __sync_fetch_and_add(write_seq, 1);
	u32 read_key = 1;
	u64 *read_seq = bpf_map_lookup_elem(&queue_state, &read_key);
	if (read_seq && seq >= *read_seq + LAUNCH_QUEUE_SIZE) {
		u32 overflow_key = 3;
		u64 *overflow = bpf_map_lookup_elem(&queue_state, &overflow_key);
		if (overflow)
			__sync_fetch_and_add(overflow, 1);
		return 0;
	}
	u32 slot = seq & (LAUNCH_QUEUE_SIZE - 1);
	bpf_map_update_elem(&launch_times, &slot, &ts_calibrated, BPF_ANY);

	return 0;
}

// Helper function to determine histogram bin based on time value
static __always_inline u32 get_hist_bin(u64 delta_ns)
{
	// Bins: 0-100ns, 100-1000ns, 1-10us, 10-100us, 100us-1ms, 1-10ms,
	// 10-100ms, 100ms-1s, >1s
	if (delta_ns < 100)           return 0;  // 0-100ns
	if (delta_ns < 1000)          return 1;  // 100ns-1us
	if (delta_ns < 10000)         return 2;  // 1-10us
	if (delta_ns < 100000)        return 3;  // 10-100us
	if (delta_ns < 1000000)       return 4;  // 100us-1ms
	if (delta_ns < 10000000)      return 5;  // 1-10ms
	if (delta_ns < 100000000)     return 6;  // 10-100ms
	if (delta_ns < 1000000000)    return 7;  // 100ms-1s
	if (delta_ns < 10000000000)   return 8;  // 1s-10s
	return 9;  // >10s
}

// GPU-side probe: one sample per selected kernel launch. The exact host-stub
// timestamps are consumed FIFO, which prevents unrelated or later launches from
// overwriting the timestamp for this kernel. Only thread (0,0,0) of block
// (0,0,0) samples, so the shared read cursor and histogram use plain (non-atomic)
// updates; the GPU verifier on this driver rejects XADD with a non-zero
// immediate.
SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe()
{
	u64 block_x, block_y, block_z;
	u64 thread_x, thread_y, thread_z;
	bpf_get_block_idx(&block_x, &block_y, &block_z);
	bpf_get_thread_idx(&thread_x, &thread_y, &thread_z);
	if (block_x || block_y || block_z || thread_x || thread_y || thread_z)
		return 0;

	u32 write_key = 0;
	u32 read_key = 1;
	u64 *write_seq = bpf_map_lookup_elem(&queue_state, &write_key);
	u64 *read_seq = bpf_map_lookup_elem(&queue_state, &read_key);
	if (!write_seq || !read_seq || *read_seq >= *write_seq)
		return 0;

	u64 seq = *read_seq;
	*read_seq = seq + 1;

	u32 slot = seq & (LAUNCH_QUEUE_SIZE - 1);
	u64 *launch_ts = bpf_map_lookup_elem(&launch_times, &slot);
	if (!launch_ts || *launch_ts == 0)
		return 0;
	u64 gpu_ts = bpf_get_globaltimer();
	u64 delta_ns = gpu_ts > *launch_ts ? gpu_ts - *launch_ts : 0;
	u32 bin = get_hist_bin(delta_ns);
	u64 *count = bpf_map_lookup_elem(&time_histogram, &bin);
	if (count)
		*count = *count + 1;

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
