/*
 * Live CUDA demo: lane-varying branch using thread_idx.
 * STRICT mode should reject this before the GPU hook is injected.
 */

#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>

static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;
static const u64 (*bpf_get_globaltimer)(void) = (void *)502;

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__live_unsafe_varying_branch(void)
{
	u64 thread_x = 0;
	u64 thread_y = 0;
	u64 thread_z = 0;

	bpf_get_thread_idx(&thread_x, &thread_y, &thread_z);
	if ((thread_x & 1) != 0) {
		(void)bpf_get_globaltimer();
	}
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
