#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503
#define BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP 1513

static const void (*ebpf_puts)(const char *) = (void *)501;
static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

#define HIST_BINS 10
#define LAUNCH_QUEUE_SIZE 4096

enum host_counter {
	HOST_LAUNCHES = 0,
	HOST_ENQUEUED = 1,
	QUEUE_OVERFLOWS = 2,
	QUEUE_UPDATE_ERRORS = 3,
	HOST_LAUNCH_CALLS = 4,
	HOST_TARGET_ERRORS = 5,
	HOST_COUNTERS = 6,
};

enum device_counter {
	DEVICE_ENTRIES = 0,
	MATCHED_SAMPLES = 1,
	QUEUE_UNDERFLOWS = 2,
	CLASSIFIED_SAMPLES = 3,
	UNCERTAIN_SAMPLES = 4,
	CLOCK_ERRORS = 5,
	DEVICE_COUNTERS = 6,
};

struct launch_sample {
	u64 host_mono_ns;
	u64 sequence;
};

struct clock_calibration {
	s64 offset_low_ns;
	s64 offset_high_ns;
	u64 uncertainty_ns;
	u64 valid;
};

struct host_target {
	u64 launch_vaddr;
	u64 kernel_vaddr;
	u64 valid;
};

// Array map to store time distribution histogram
// Each element represents count for a time range
struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, HIST_BINS);
	__type(key, u32);
	__type(value, u64);
} time_histogram SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP);
	__uint(max_entries, LAUNCH_QUEUE_SIZE);
	__type(key, u32);
	__type(value, struct launch_sample);
} launch_times SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, HOST_COUNTERS);
	__type(key, u32);
	__type(value, u64);
} host_state SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, DEVICE_COUNTERS);
	__type(key, u32);
	__type(value, u64);
} device_state SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct clock_calibration);
} clock_calibration SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct host_target);
} host_target SEC(".maps");

static __always_inline void increment_host_counter(u32 key)
{
	u64 *value = bpf_map_lookup_elem(&host_state, &key);
	if (value)
		__sync_fetch_and_add(value, 1);
}

static __always_inline void increment_device_counter(u32 key)
{
	u64 *value = bpf_map_lookup_elem(&device_state, &key);
	if (value)
		__sync_fetch_and_add(value, 1);
}

/*
 * The userspace loader attaches this uprobe to cudaLaunchKernel's PLT entry in
 * the target ELF.  Both addresses below are virtual addresses from that same
 * ELF, so the live PLT address supplies the load bias.  Only an exact arg0
 * match is allowed to enter the FIFO.
 */
SEC("uprobe")
int BPF_KPROBE(uprobe_cuda_launch, const void *func)
{
	u32 target_key = 0;
	struct host_target *target;
	u64 launch_ip, delta, expected;

	increment_host_counter(HOST_LAUNCH_CALLS);
	target = bpf_map_lookup_elem(&host_target, &target_key);
	launch_ip = bpf_get_func_ip(ctx);
	if (!target || !target->valid || !launch_ip) {
		increment_host_counter(HOST_TARGET_ERRORS);
		return 0;
	}
	if (target->kernel_vaddr >= target->launch_vaddr) {
		delta = target->kernel_vaddr - target->launch_vaddr;
		if (delta > ~launch_ip) {
			increment_host_counter(HOST_TARGET_ERRORS);
			return 0;
		}
		expected = launch_ip + delta;
	} else {
		delta = target->launch_vaddr - target->kernel_vaddr;
		if (launch_ip < delta) {
			increment_host_counter(HOST_TARGET_ERRORS);
			return 0;
		}
		expected = launch_ip - delta;
	}
	if ((u64)func != expected)
		return 0;

	u32 host_key = HOST_LAUNCHES;
	u64 *host_launches = bpf_map_lookup_elem(&host_state, &host_key);
	if (!host_launches)
		return 0;

	u64 seq = __sync_fetch_and_add(host_launches, 1);
	/* This trace is deliberately bounded rather than silently overwriting. */
	if (seq >= LAUNCH_QUEUE_SIZE) {
		increment_host_counter(QUEUE_OVERFLOWS);
		return 0;
	}

	u32 slot = seq & (LAUNCH_QUEUE_SIZE - 1);
	struct launch_sample sample = {
		.host_mono_ns = bpf_ktime_get_ns(),
		.sequence = seq + 1,
	};
	if (bpf_map_update_elem(&launch_times, &slot, &sample, BPF_ANY)) {
		increment_host_counter(QUEUE_UPDATE_ERRORS);
		return 0;
	}
	increment_host_counter(HOST_ENQUEUED);

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

// GPU-side probe: one sample per selected kernel launch. The build-time target
// must name the same wrapper supplied to the userspace loader. FIFO pairing is
// fail-closed and requires host-launch order to equal device-entry order.
#ifndef LAUNCHLATE_TARGET_SYMBOL
SEC("kprobe/_Z9vectorAddPKfS0_Pf")
#else
SEC("kprobe/" LAUNCHLATE_TARGET_SYMBOL)
#endif
int cuda__probe()
{
	u64 block_x, block_y, block_z;
	u64 thread_x, thread_y, thread_z;
	bpf_get_block_idx(&block_x, &block_y, &block_z);
	bpf_get_thread_idx(&thread_x, &thread_y, &thread_z);
	if (block_x || block_y || block_z || thread_x || thread_y || thread_z)
		return 0;

	u32 device_key = DEVICE_ENTRIES;
	u64 *device_entries = bpf_map_lookup_elem(&device_state, &device_key);
	if (!device_entries)
		return 0;
	u64 seq = __sync_fetch_and_add(device_entries, 1);
	if (seq >= LAUNCH_QUEUE_SIZE) {
		increment_device_counter(QUEUE_UNDERFLOWS);
		return 0;
	}
	u32 slot = seq & (LAUNCH_QUEUE_SIZE - 1);
	struct launch_sample *sample = bpf_map_lookup_elem(&launch_times, &slot);
	if (!sample || sample->sequence != seq + 1 || !sample->host_mono_ns) {
		increment_device_counter(QUEUE_UNDERFLOWS);
		return 0;
	}
	increment_device_counter(MATCHED_SAMPLES);

	u32 calibration_key = 0;
	struct clock_calibration *calibration =
		bpf_map_lookup_elem(&clock_calibration, &calibration_key);
	if (!calibration || !calibration->valid ||
	    calibration->offset_low_ns > calibration->offset_high_ns) {
		increment_device_counter(CLOCK_ERRORS);
		return 0;
	}

	u64 gpu_ts = bpf_get_globaltimer();
	if (gpu_ts > 0x7fffffffffffffffULL ||
	    sample->host_mono_ns > 0x7fffffffffffffffULL) {
		increment_device_counter(CLOCK_ERRORS);
		return 0;
	}

	/*
	 * The calibration kernel establishes an interval for
	 *   GPU globaltimer - CLOCK_MONOTONIC.
	 * Keep a sample out of the histogram unless its entire possible latency
	 * interval falls in one bin.  This avoids silently clamping negative
	 * deltas or pretending the midpoint is exact.
	 */
	s64 observed_ns = (s64)gpu_ts - (s64)sample->host_mono_ns;
	s64 latency_low_ns = observed_ns - calibration->offset_high_ns;
	s64 latency_high_ns = observed_ns - calibration->offset_low_ns;
	if (latency_high_ns < 0) {
		increment_device_counter(CLOCK_ERRORS);
		return 0;
	}
	if (latency_low_ns < 0) {
		increment_device_counter(UNCERTAIN_SAMPLES);
		return 0;
	}

	u32 low_bin = get_hist_bin((u64)latency_low_ns);
	u32 high_bin = get_hist_bin((u64)latency_high_ns);
	if (low_bin != high_bin) {
		increment_device_counter(UNCERTAIN_SAMPLES);
		return 0;
	}
	u32 bin = low_bin;
	u64 *count = bpf_map_lookup_elem(&time_histogram, &bin);
	if (count) {
		__sync_fetch_and_add(count, 1);
		increment_device_counter(CLASSIFIED_SAMPLES);
	} else {
		increment_device_counter(CLOCK_ERRORS);
	}

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
