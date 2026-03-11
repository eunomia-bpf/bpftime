/**
 * multi_gpu_probe.bpf.c - GPU-side eBPF probe for load balance analysis
 *
 * Instruments the vectorAdd kernel with kprobe/kretprobe to provide
 * GPU-internal per-block execution timing. This complements the host-side
 * CUDA event timing by giving fine-grained block-level latency distribution.
 *
 * What this probe measures (from INSIDE the GPU):
 *   - Per-block kernel execution duration via globaltimer
 *   - Total kernel invocation count (incremented by block 0)
 *   - Timing histogram: count of blocks in each latency bucket
 *   - Min and max observed block duration
 *
 * These GPU-internal metrics are invisible to CUDA events and require
 * either NVBit or bpftime's eBPF GPU attach to obtain.
 */
#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

// --- Maps ---

// Per-block entry timestamps (transient, cleared after each kretprobe)
// Key is (device_ordinal << 20 | block_id) to avoid cross-GPU collisions
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 65536);
	__type(key, u32);    // (device_ordinal << 20) | block_id
	__type(value, u64);  // globaltimer value at entry
} start_ts SEC(".maps");

// Total kernel invocation count (only block 0 increments)
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 8);
	__type(key, u32);    // always 0
	__type(value, u64);  // count
} invoke_count SEC(".maps");

// Timing histogram: key = latency bucket index, value = count
// Bucket 0: 0-1us, 1: 1-10us, 2: 10-100us, 3: 100us-1ms,
// 4: 1-10ms, 5: 10-100ms, 6: 100ms+
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 16);
	__type(key, u32);    // bucket index (0-6)
	__type(value, u64);  // count of blocks in this bucket
} latency_hist SEC(".maps");

// Aggregated stats: key=stat_id, value=nanoseconds
// stat_id 0: sum of all block durations (for computing average)
// stat_id 1: min block duration
// stat_id 2: max block duration
// stat_id 3: total block count
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 8);
	__type(key, u32);
	__type(value, u64);
} duration_stats SEC(".maps");

// Per-GPU block stats: key = device_ordinal (set per-module by bpftime)
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 16);
	__type(key, u32);    // device ordinal
	__type(value, u64);  // sum of block durations (ns)
} per_gpu_time SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 16);
	__type(key, u32);    // device ordinal
	__type(value, u64);  // block count
} per_gpu_count SEC(".maps");

// --- GPU helpers ---
static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;
static const u64 (*bpf_get_grid_dim)(u64 *x, u64 *y, u64 *z) = (void *)508;
static const u64 (*bpf_get_device_ordinal)(void) = (void *)512;

// Classify latency into histogram bucket
static __always_inline u32 latency_bucket(u64 ns)
{
	if (ns < 1000)          return 0; // <1us
	if (ns < 10000)         return 1; // 1-10us
	if (ns < 100000)        return 2; // 10-100us
	if (ns < 1000000)       return 3; // 100us-1ms
	if (ns < 10000000)      return 4; // 1-10ms
	if (ns < 100000000)     return 5; // 10-100ms
	return 6;                         // 100ms+
}

/*
 * Kprobe: fires at vectorAdd entry on every GPU that runs this kernel.
 * Mangled name: void vectorAdd(const float*, const float*, float*, int)
 */
SEC("kprobe/_Z9vectorAddPKfS0_Pfi")
int cuda__vec_add_entry()
{
	u64 bx, by, bz;
	bpf_get_block_idx(&bx, &by, &bz);
	u32 dev_ord = (u32)bpf_get_device_ordinal();
	// Compound key: device_ordinal << 20 | block_id (supports up to 1M blocks)
	u32 ts_key = (dev_ord << 20) | ((u32)bx & 0xFFFFF);

	u64 ts = bpf_get_globaltimer();
	bpf_map_update_elem(&start_ts, &ts_key, &ts, BPF_ANY);

	// Block 0 increments the global invocation counter
	if ((u32)bx == 0) {
		u32 key = 0;
		u64 one = 1;
		u64 *cnt = bpf_map_lookup_elem(&invoke_count, &key);
		if (cnt) {
			__atomic_add_fetch(cnt, 1, __ATOMIC_SEQ_CST);
		} else {
			bpf_map_update_elem(&invoke_count, &key, &one,
					    BPF_NOEXIST);
		}
	}

	return 0;
}

/*
 * Kretprobe: fires at vectorAdd exit. Computes per-block duration and
 * updates histogram + aggregate stats.
 */
SEC("kretprobe/_Z9vectorAddPKfS0_Pfi")
int cuda__vec_add_exit()
{
	u64 bx, by, bz;
	bpf_get_block_idx(&bx, &by, &bz);
	u32 dev_ord = (u32)bpf_get_device_ordinal();
	u32 ts_key = (dev_ord << 20) | ((u32)bx & 0xFFFFF);

	u64 *tsp = bpf_map_lookup_elem(&start_ts, &ts_key);
	if (!tsp)
		return 0;

	u64 delta = bpf_get_globaltimer() - *tsp;
	bpf_map_delete_elem(&start_ts, &ts_key);

	// Update latency histogram
	u32 bucket = latency_bucket(delta);
	u64 one = 1;
	u64 *hist_val = bpf_map_lookup_elem(&latency_hist, &bucket);
	if (hist_val) {
		__atomic_add_fetch(hist_val, 1, __ATOMIC_SEQ_CST);
	} else {
		bpf_map_update_elem(&latency_hist, &bucket, &one,
				    BPF_NOEXIST);
	}

	// Update aggregate stats
	// stat 0: sum
	u32 stat_sum = 0;
	u64 *sum_p = bpf_map_lookup_elem(&duration_stats, &stat_sum);
	if (sum_p) {
		__atomic_add_fetch(sum_p, delta, __ATOMIC_SEQ_CST);
	} else {
		bpf_map_update_elem(&duration_stats, &stat_sum, &delta,
				    BPF_NOEXIST);
	}

	// stat 1: min (relaxed - may miss races, acceptable for monitoring)
	u32 stat_min = 1;
	u64 *min_p = bpf_map_lookup_elem(&duration_stats, &stat_min);
	if (min_p) {
		if (delta < *min_p)
			bpf_map_update_elem(&duration_stats, &stat_min, &delta,
					    BPF_ANY);
	} else {
		bpf_map_update_elem(&duration_stats, &stat_min, &delta,
				    BPF_NOEXIST);
	}

	// stat 2: max
	u32 stat_max = 2;
	u64 *max_p = bpf_map_lookup_elem(&duration_stats, &stat_max);
	if (max_p) {
		if (delta > *max_p)
			bpf_map_update_elem(&duration_stats, &stat_max, &delta,
					    BPF_ANY);
	} else {
		bpf_map_update_elem(&duration_stats, &stat_max, &delta,
				    BPF_NOEXIST);
	}

	// stat 3: total block count
	u32 stat_cnt = 3;
	u64 *cnt_p = bpf_map_lookup_elem(&duration_stats, &stat_cnt);
	if (cnt_p) {
		__atomic_add_fetch(cnt_p, 1, __ATOMIC_SEQ_CST);
	} else {
		bpf_map_update_elem(&duration_stats, &stat_cnt, &one,
				    BPF_NOEXIST);
	}

	// Per-GPU stats: use device ordinal (set per-module by bpftime)
	u64 *gtime = bpf_map_lookup_elem(&per_gpu_time, &dev_ord);
	if (gtime) {
		__atomic_add_fetch(gtime, delta, __ATOMIC_SEQ_CST);
	} else {
		bpf_map_update_elem(&per_gpu_time, &dev_ord, &delta,
				    BPF_NOEXIST);
	}

	u64 *gcnt = bpf_map_lookup_elem(&per_gpu_count, &dev_ord);
	if (gcnt) {
		__atomic_add_fetch(gcnt, 1, __ATOMIC_SEQ_CST);
	} else {
		bpf_map_update_elem(&per_gpu_count, &dev_ord, &one,
				    BPF_NOEXIST);
	}

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
