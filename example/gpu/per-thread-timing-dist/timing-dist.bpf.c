#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

typedef unsigned long long u64;
typedef unsigned int u32;

// Map to store entry timestamps
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 4096);
	__type(key, u32);
	__type(value, u64);
} start_ts SEC(".maps");

// Histogram map to store timing distribution
// The key is the log2 of the duration in nanoseconds
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 64); // To store log2 of durations
	__type(key, u32);
	__type(value, u64);
} timing_dist SEC(".maps");

struct summary_t {
	u64 min;
	u64 max;
	u64 sum;
	u64 count;
};

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct summary_t);
} summary_map SEC(".maps");

static __always_inline u64 log2l(u64 v)
{
	u64 r = 0;
	if (v == 0)
		return 0;
	if (v & 0xFFFFFFFF00000000ULL) {
		v >>= 32;
		r |= 32;
	}
	if (v & 0xFFFF0000) {
		v >>= 16;
		r |= 16;
	}
	if (v & 0xFF00) {
		v >>= 8;
		r |= 8;
	}
	if (v & 0xF0) {
		v >>= 4;
		r |= 4;
	}
	if (v & 0xC) {
		v >>= 2;
		r |= 2;
	}
	if (v & 0x2) {
		v >>= 1;
		r |= 1;
	}
	return r;
}

static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

SEC("kprobe/timed_work_kernel")
int BPF_KPROBE(timed_work_kernel_enter)
{
	u64 block_x, block_y, block_z;
	bpf_get_block_idx(&block_x, &block_y, &block_z);
	u64 block_dim_x, block_dim_y, block_dim_z;
	bpf_get_block_dim(&block_dim_x, &block_dim_y, &block_dim_z);
	u64 thread_x, thread_y, thread_z;
	bpf_get_thread_idx(&thread_x, &thread_y, &thread_z);

	u32 global_thread_id = block_x * block_dim_x + thread_x;

	u64 ts = bpf_get_globaltimer();
	bpf_map_update_elem(&start_ts, &global_thread_id, &ts, BPF_ANY);

	return 0;
}

SEC("kretprobe/timed_work_kernel")
int BPF_KRETPROBE(timed_work_kernel_exit)
{
	u64 block_x, block_y, block_z;
	bpf_get_block_idx(&block_x, &block_y, &block_z);
	u64 block_dim_x, block_dim_y, block_dim_z;
	bpf_get_block_dim(&block_dim_x, &block_dim_y, &block_dim_z);
	u64 thread_x, thread_y, thread_z;
	bpf_get_thread_idx(&thread_x, &thread_y, &thread_z);

	u32 global_thread_id = block_x * block_dim_x + thread_x;

	u64 *tsp = bpf_map_lookup_elem(&start_ts, &global_thread_id);
	if (tsp) {
		u64 delta = bpf_get_globaltimer() - *tsp;
		bpf_map_delete_elem(&start_ts, &global_thread_id);

		u64 slot = log2l(delta);
		if (slot >= 64)
			slot = 63;

		u64 *count = (u64 *)bpf_map_lookup_elem(&timing_dist, &slot);
		if (count) {
			__atomic_add_fetch(count, 1, __ATOMIC_RELAXED);
		}

		u32 key = 0;
		struct summary_t *s = bpf_map_lookup_elem(&summary_map, &key);
		if (s) {
			__atomic_add_fetch(&s->sum, delta, __ATOMIC_RELAXED);
			__atomic_add_fetch(&s->count, 1, __ATOMIC_RELAXED);

			u64 old_min = s->min;
			if (delta < old_min) {
				__sync_val_compare_and_swap(&s->min, old_min,
							    delta);
			}
			u64 old_max = s->max;
			if (delta > old_max) {
				__sync_val_compare_and_swap(&s->max, old_max,
							    delta);
			}
		}
	}
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
