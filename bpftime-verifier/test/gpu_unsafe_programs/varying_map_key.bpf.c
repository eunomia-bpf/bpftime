/*
 * Safety property: Violates map-update key uniformity. The selected key slot
 * and the key value itself are derived from bpf_get_lane_id() (helper 511).
 * Expected verifier result: REJECT.
 * Why this matters for GPU execution: lane-varying map updates fragment a
 * single logical hook execution into per-lane side effects, increasing
 * contention and breaking warp-uniform update assumptions.
 */

#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_HASH_MAP 1501

struct {
	__uint(type, BPF_MAP_TYPE_GPU_HASH_MAP);
	__uint(max_entries, 32);
	__type(key, u32);
	__type(value, u64);
} per_lane_updates SEC(".maps");

static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_lane_id)(void) = (void *)511;

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__varying_map_key()
{
	u32 lane_keys[4] = { 0, 1, 2, 3 };
	u64 lane_id = bpf_get_lane_id();
	u32 *keyp = &lane_keys[lane_id & 3];
	u64 value = bpf_get_globaltimer();

	bpf_map_update_elem(&per_lane_updates, keyp, &value, BPF_ANY);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
