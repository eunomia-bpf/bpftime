/**
 * multi_gpu_probe.bpf.c - eBPF probe for multi-GPU kernel tracing
 *
 * This probe instruments the vectorAdd kernel running on multiple GPUs.
 * It tracks per-block execution timing and call counts, demonstrating
 * that bpftime can trace the same kernel across different GPU devices.
 */
#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

// Map to store entry timestamps per block
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 4096);
	__type(key, u32);
	__type(value, u64);
} start_ts SEC(".maps");

// Map to store kernel invocation counts
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 64);
	__type(key, u32);
	__type(value, u64);
} call_count SEC(".maps");

// Map to store total execution time per block
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 4096);
	__type(key, u32);
	__type(value, u64);
} total_time_ns SEC(".maps");

static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

/*
 * Kprobe on vectorAdd - fires when the kernel starts on any GPU.
 * The mangled name matches the CUDA kernel signature:
 *   __global__ void vectorAdd(const float*, const float*, float*, int)
 */
SEC("kprobe/_Z9vectorAddPKfS0_Pfi")
int cuda__vec_add_entry()
{
	u64 x, y, z;
	bpf_get_block_idx(&x, &y, &z);
	u32 block_id = (u32)x;

	// Record entry timestamp
	u64 ts = bpf_get_globaltimer();
	bpf_map_update_elem(&start_ts, &block_id, &ts, BPF_ANY);

	// Increment call count (use block 0 as representative)
	if (block_id == 0) {
		u64 one = 1;
		u64 *cnt = bpf_map_lookup_elem(&call_count, &block_id);
		if (cnt) {
			__atomic_add_fetch(cnt, 1, __ATOMIC_SEQ_CST);
		} else {
			bpf_map_update_elem(&call_count, &block_id, &one,
					    BPF_NOEXIST);
		}
	}

	bpf_printk("vectorAdd entry: block=%u, ts=%lu\n", block_id, ts);
	return 0;
}

/*
 * Kretprobe on vectorAdd - fires when the kernel finishes on any GPU.
 */
SEC("kretprobe/_Z9vectorAddPKfS0_Pfi")
int cuda__vec_add_exit()
{
	u64 x, y, z;
	bpf_get_block_idx(&x, &y, &z);
	u32 block_id = (u32)x;

	u64 *tsp = bpf_map_lookup_elem(&start_ts, &block_id);
	if (tsp) {
		u64 delta = bpf_get_globaltimer() - *tsp;
		bpf_map_delete_elem(&start_ts, &block_id);

		// Accumulate total time
		u64 *total = bpf_map_lookup_elem(&total_time_ns, &block_id);
		if (total) {
			*total += delta;
			bpf_map_update_elem(&total_time_ns, &block_id, total,
					    BPF_EXIST);
		} else {
			bpf_map_update_elem(&total_time_ns, &block_id, &delta,
					    BPF_NOEXIST);
		}

		bpf_printk("vectorAdd exit: block=%u, duration=%lu ns\n",
			   block_id, delta);
	}

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
