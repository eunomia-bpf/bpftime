/*
 * Safety property: Satisfies warp-uniform control flow. The branch predicate
 * comes from bpf_get_block_idx() (helper 503), which is uniform within a warp.
 * Expected verifier result: PASS.
 * Why this matters for GPU execution: block-level decisions preserve warp
 * convergence while still allowing the hook to specialize behavior by block.
 */

#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, 2);
	__type(key, u32);
	__type(value, u64);
} block_buckets SEC(".maps");

static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__safe_block_idx_branch()
{
	u64 block_x = 0, block_y = 0, block_z = 0;
	u32 key = 0;
	u64 value = bpf_get_globaltimer();

	bpf_get_block_idx(&block_x, &block_y, &block_z);

	if (block_x & 1)
		key = 1;

	bpf_map_update_elem(&block_buckets, &key, &value, BPF_ANY);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
