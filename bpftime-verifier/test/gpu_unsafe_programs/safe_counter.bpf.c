/*
 * Safety property: Satisfies warp-uniform control flow and uniform atomic side
 * effects. Branches depend only on bpf_get_warp_id() (helper 510) or on the
 * result of a fixed-key lookup, both of which are uniform.
 * Expected verifier result: PASS.
 * Why this matters for GPU execution: this is the intended shape for a
 * lightweight GPU hook, where all lanes in a warp agree on control flow and
 * shared-state updates target a single uniform location.
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
} safe_counter_map SEC(".maps");

static const u64 (*bpf_get_warp_id)(void) = (void *)510;

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__safe_counter()
{
	u32 key = 0;
	u64 initial = 1;
	u64 warp_id = bpf_get_warp_id();
	u64 *counter;

	/* warp_id is uniform within the executing warp. */
	if (warp_id != 0)
		return 0;

	counter = bpf_map_lookup_elem(&safe_counter_map, &key);
	if (counter) {
		__sync_fetch_and_add(counter, 1);
	} else {
		bpf_map_update_elem(&safe_counter_map, &key, &initial,
				    BPF_ANY);
	}

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
