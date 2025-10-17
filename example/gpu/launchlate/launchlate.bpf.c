#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503

static const void (*ebpf_puts)(const char *) = (void *)501;
static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

// Define histogram bins for time distribution
// Bins: 0-100ns, 100-1000ns, 1-10us, 10-100us, 100us-1ms, 1-10ms, 10-100ms, 100ms-1s, >1s
#define HIST_BINS 10

// Array map to store time distribution histogram
// Each element represents count for a time range
struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, HIST_BINS);
	__type(key, u32);
	__type(value, u64);
} time_histogram SEC(".maps");

// Map to store the last uprobe timestamp for calculating latency
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 32);
	__type(key, u32);
	__type(value, u64);
} last_uprobe_time SEC(".maps");

// Map to store clock calibration offset
// Index 0: realtime_ns - monotonic_ns offset
struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 32);
	__type(key, u32);
	__type(value, s64);
} clock_offset SEC(".maps");

// Uprobe on cudaLaunchKernel - tracks when kernels are launched from CPU
// Note: actual uprobe target is configured at runtime via bpf_program__attach_uprobe()
SEC("uprobe")
int BPF_KPROBE(uprobe_cuda_launch, const void *func, u64 gridDim, u64 blockDim)
{
	u64 ts_mono = bpf_ktime_get_ns();
	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u32 key = 0;
	s64 *offset_ptr;
	u64 ts_calibrated = ts_mono;

	bpf_printk("CPU: cudaLaunchKernel called at ts=%lu ns (calibrated=%lu), pid=%u\n",
		   ts_mono, ts_calibrated, pid);

	// Apply clock offset to convert monotonic time to approximate realtime
	offset_ptr = bpf_map_lookup_elem(&clock_offset, &key);
	if (offset_ptr) {
		ts_calibrated = ts_mono + *offset_ptr;
	}
	// bpf_printk("okk");
	// Store the timestamp for latency calculation
	bpf_map_update_elem(&last_uprobe_time, &key, &ts_calibrated, BPF_ANY);

	return 0;
}

// Helper function to determine histogram bin based on time value
static __always_inline u32 get_hist_bin(u64 delta_ns)
{
	// Bins: 0-100ns, 100-1000ns, 1-10us, 10-100us, 100us-1ms, 1-10ms, 10-100ms, 100ms-1s, >1s
	if (delta_ns < 100)           return 0;  // 0-100ns
	if (delta_ns < 1000)          return 1;  // 100ns-1us
	if (delta_ns < 10000)         return 2;  // 1-10us
	if (delta_ns < 100000)        return 3;  // 10-100us
	if (delta_ns < 1000000)       return 4;  // 100us-1ms
	if (delta_ns < 10000000)      return 5;  // 1-10ms
	if (delta_ns < 100000000)     return 6;  // 10-100ms
	if (delta_ns < 1000000000)    return 7;  // 100ms-1s
	if (delta_ns < 10000000000)   return 8;  // 1s-10s
	return 9;  // >10s
}

// GPU-side probe - tracks when kernel actually executes on GPU
SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe()
{
	u64 gpu_ts = bpf_get_globaltimer();
	u32 key = 0;
	u64 *uprobe_ts;
	u64 delta_ns = 0;

	// Get the last uprobe timestamp
	uprobe_ts = bpf_map_lookup_elem(&last_uprobe_time, &key);
	if (uprobe_ts && *uprobe_ts > 0) {
		// Calculate time difference between uprobe and GPU execution
		if (gpu_ts > *uprobe_ts) {
			delta_ns = gpu_ts - *uprobe_ts;
		}

		// Determine which histogram bin this latency falls into
		u32 bin = get_hist_bin(delta_ns);

		// Update the histogram count for this bin
		u64 *count = bpf_map_lookup_elem(&time_histogram, &bin);
		if (count) {
			*count += 1;
		} else {
			u64 one = 1;
			bpf_map_update_elem(&time_histogram, &bin, &one, BPF_NOEXIST);
		}

		bpf_printk("GPU: Kernel executing, latency=%lu ns (bin=%u), gpu_ts=%lu\n",
			   delta_ns, bin, gpu_ts);
	} else {
		bpf_printk("GPU: Kernel executing but no uprobe timestamp found, gpu_ts=%lu\n", gpu_ts);
	}

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
