/*
 * Safety property: Violates the requirement that atomic side effects use a
 * warp-uniform target address. The atomic slot is derived from thread_idx.
 * Expected verifier result: REJECT.
 * Why this matters for GPU execution: lane-varying atomic destinations create
 * per-lane side effects inside a single warp and defeat SIMT-aware reasoning
 * about shared-state updates.
 */

#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503

struct counter_table {
	u64 slots[32];
};

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct counter_table);
} counters SEC(".maps");

static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__varying_atomic()
{
	u32 key = 0;
	u64 thread_x = 0, thread_y = 0, thread_z = 0;
	struct counter_table *table = bpf_map_lookup_elem(&counters, &key);
	u64 *slot;

	if (!table)
		return 0;

	bpf_get_thread_idx(&thread_x, &thread_y, &thread_z);

	slot = &table->slots[thread_x & 31];
	__sync_fetch_and_add(slot, 1);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
