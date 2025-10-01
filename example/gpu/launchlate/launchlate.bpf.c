#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

static const void (*ebpf_puts)(const char *) = (void *)501;
static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

// Map to store call count
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, u32);
	__type(value, u64);
} call_count SEC(".maps");

// Map to store clock calibration offset
// Index 0: realtime_ns - monotonic_ns offset
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, s64);
} clock_offset SEC(".maps");

// Uprobe on cudaLaunchKernel - tracks when kernels are launched from CPU
SEC("uprobe/example/gpu/launchlate/vec_add:_Z16cudaLaunchKernelIcE9cudaErrorPT_4dim3S3_PPvmP11CUstream_st")
int BPF_KPROBE(uprobe_cuda_launch, const void *func, u64 gridDim, u64 blockDim)
{
	u64 ts_mono = bpf_ktime_get_ns();
	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u32 offset_key = 0;
	s64 *offset_ptr;
	u64 ts_calibrated = ts_mono;

	// Apply clock offset to convert monotonic time to approximate realtime
	offset_ptr = bpf_map_lookup_elem(&clock_offset, &offset_key);
	if (offset_ptr) {
		ts_calibrated = ts_mono + *offset_ptr;
	}

	bpf_printk("CPU: cudaLaunchKernel called at ts=%lu ns (calibrated=%lu), pid=%u\n",
		   ts_mono, ts_calibrated, pid);

	return 0;
}

// GPU-side probe - tracks when kernel actually executes on GPU
SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int probe__cuda()
{
	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u64 ts = bpf_get_globaltimer();
	u64 x, y, z;
	bpf_get_block_idx(&x, &y, &z);

	// Increment call count
	u64 one = 1;
	u64 *cnt = bpf_map_lookup_elem(&call_count, &pid);
	if (cnt) {
		__atomic_add_fetch(cnt, 1, __ATOMIC_SEQ_CST);
	} else {
		bpf_map_update_elem(&call_count, &pid, &one, BPF_NOEXIST);
	}

	bpf_printk("GPU: Kernel executing x=%lu, ts=%lu\n", x, ts);

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
