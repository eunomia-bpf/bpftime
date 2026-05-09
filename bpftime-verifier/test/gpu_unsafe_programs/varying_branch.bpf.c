/*
 * Safety property: Violates warp-uniform control flow. The branch predicate
 * comes from bpf_get_thread_idx() (helper 505), which is lane-varying.
 * Expected verifier result: REJECT.
 * Why this matters for GPU execution: different lanes in the same warp can
 * take different paths, forcing warp serialization and breaking SIMT-aware
 * assumptions about uniform control flow in the injected eBPF hook.
 */

#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_HASH_MAP 1501

struct {
	__uint(type, BPF_MAP_TYPE_GPU_HASH_MAP);
	__uint(max_entries, 4);
	__type(key, u32);
	__type(value, u64);
} divergence_paths SEC(".maps");

static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__varying_branch()
{
	u64 thread_x = 0, thread_y = 0, thread_z = 0;
	u64 stamp = bpf_get_globaltimer();
	u32 key = 0;

	bpf_get_thread_idx(&thread_x, &thread_y, &thread_z);

	if ((thread_x & 3) == 0) {
		key = 0;
	} else if ((thread_x & 3) == 1) {
		key = 1;
	} else {
		key = 2;
	}

	bpf_map_update_elem(&divergence_paths, &key, &stamp, BPF_ANY);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
