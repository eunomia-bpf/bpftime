/*
 * Live CUDA demo: warp-safe control flow using block_idx only.
 * This is intentionally map-free so the runtime demonstration isolates
 * SIMT-safety behavior from map relocation issues in heavier examples.
 */

#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>

static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_globaltimer)(void) = (void *)502;

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__live_safe_block_idx(void)
{
	u64 block_x = 0;
	u64 block_y = 0;
	u64 block_z = 0;

	bpf_get_block_idx(&block_x, &block_y, &block_z);
	if ((block_x & 1) != 0) {
		(void)bpf_get_globaltimer();
	}
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
