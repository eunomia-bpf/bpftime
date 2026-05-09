/*
 * Safety property: Violates the "no prohibited GPU synchronization helpers"
 * rule by calling bpf_gpu_membar() (helper 506).
 * Expected verifier result: REJECT.
 * Why this matters for GPU execution: fence and barrier style helpers can
 * introduce ordering or waiting behavior that is unsafe in SIMT execution and
 * must be rejected conservatively by the verifier.
 */

#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} fence_state SEC(".maps");

static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const long (*bpf_gpu_membar)(void) = (void *)506;

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__prohibited_helper()
{
	u64 block_x = 0, block_y = 0, block_z = 0;
	u32 key = 0;
	u64 value = 1;

	bpf_get_block_idx(&block_x, &block_y, &block_z);
	if (block_x == 0)
		value = 2;

	bpf_map_update_elem(&fence_state, &key, &value, BPF_ANY);

	/* Explicitly prohibited in SIMT-aware verification. */
	bpf_gpu_membar();
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
