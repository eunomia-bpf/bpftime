// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
/*
 * host_map_test.bpf.c - Test BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP and
 *                       BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP
 *
 * This example demonstrates how to use Host-memory-backed GPU maps:
 * - PERGPUTD_ARRAY_HOST_MAP: Per-GPU-thread storage in Host memory
 * - GPU_ARRAY_HOST_MAP: Shared storage in Host memory (all threads share)
 */

#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

// Map type IDs (from bpftime_shm.hpp)
// GPU_MAP_OFFSET = 1500, BPF_MAP_TYPE_ARRAY = 2
// BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP = 1500 + 2 + 10 = 1512
// BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP = 1500 + 2 + 11 = 1513
#define BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP 1512
#define BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP 1513

// Default number of entries for host maps (can be overridden)
#ifndef HOST_MAP_MAX_ENTRIES
#define HOST_MAP_MAX_ENTRIES 10
#endif

// GPU helper functions
static const void (*ebpf_puts)(const char *) = (void *)501;
static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

/*
 * Per-GPU-Thread Array Host Map
 *
 * Each GPU thread has its own independent storage.
 * Storage is in Host memory, accessible by both GPU and Host.
 *
 * Memory layout: max_entries * value_size * thread_count bytes
 * - When GPU thread N writes to key K, it writes to its own slot
 * - Host can read all thread slots via a single lookup (returns array)
 */
struct {
	__uint(type, BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP);
	__uint(max_entries, HOST_MAP_MAX_ENTRIES);
	__type(key, u32);
	__type(value, u64);
} perthread_counter SEC(".maps");

/*
 * GPU Array Host Map (Shared)
 *
 * All GPU threads share the same storage (single copy).
 * Storage is in Host memory, accessible by both GPU and Host.
 *
 * Memory layout: max_entries * value_size bytes
 * - All threads see and modify the same values
 * - Last writer wins (non-atomic)
 * - Use atomic operations if accurate counting is needed
 */
struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP);
	__uint(max_entries, HOST_MAP_MAX_ENTRIES);
	__type(key, u32);
	__type(value, u64);
} shared_counter SEC(".maps");

/*
 * Store timestamp for each thread to calculate execution time
 */
struct {
	__uint(type, BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} thread_timestamp SEC(".maps");

/*
 * Kernel entry probe
 *
 * - Record entry timestamp per thread
 * - Increment shared counter (demonstrates contention)
 */
SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe()
{
	u64 tid_x, tid_y, tid_z;
	bpf_get_thread_idx(&tid_x, &tid_y, &tid_z);

	// Store entry timestamp for this thread
	u32 key = 0;
	u64 ts = bpf_get_globaltimer();
	bpf_map_update_elem(&thread_timestamp, &key, &ts, (u64)BPF_ANY);

	// Increment shared counter (key based on thread index mod HOST_MAP_MAX_ENTRIES)
	u32 shared_key = (u32)(tid_x % HOST_MAP_MAX_ENTRIES);
	u64 *shared_val = bpf_map_lookup_elem(&shared_counter, &shared_key);
	if (shared_val) {
		u64 newv = *shared_val + 1;
		bpf_map_update_elem(&shared_counter, &shared_key, &newv, (u64)BPF_ANY);
	} else {
		u64 one = 1;
		bpf_map_update_elem(&shared_counter, &shared_key, &one, (u64)BPF_ANY);
	}

	bpf_printk("Thread (%lu,0,0) entered kernel, ts=%lu\n", tid_x, ts);
	return 0;
}

/*
 * Kernel exit probe
 *
 * - Calculate execution time per thread
 * - Store per-thread counter (each thread has isolated storage)
 * - Store execution time in perthread map
 */
SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int cuda__retprobe()
{
	u64 tid_x, tid_y, tid_z;
	bpf_get_thread_idx(&tid_x, &tid_y, &tid_z);

	// Get entry timestamp for this thread
	u32 key = 0;
	u64 *start_ts = bpf_map_lookup_elem(&thread_timestamp, &key);
	u64 duration = 0;

	if (start_ts && *start_ts > 0) {
		u64 end_ts = bpf_get_globaltimer();
		duration = end_ts - *start_ts;
	}

	// Increment per-thread counter (key 0: call count)
	u32 count_key = 0;
	u64 *cnt = bpf_map_lookup_elem(&perthread_counter, &count_key);
	if (cnt) {
		u64 newv = *cnt + 1;
		bpf_map_update_elem(&perthread_counter, &count_key, &newv, (u64)BPF_ANY);
	} else {
		u64 one = 1;
		bpf_map_update_elem(&perthread_counter, &count_key, &one, (u64)BPF_ANY);
	}

	// Store execution time per thread (key 1: last execution time)
	u32 time_key = 1;
	bpf_map_update_elem(&perthread_counter, &time_key, &duration, (u64)BPF_ANY);

	// Store thread ID in key 2 (for verification)
	u32 tid_key = 2;
	bpf_map_update_elem(&perthread_counter, &tid_key, &tid_x, (u64)BPF_ANY);

	bpf_printk("Thread (%lu,0,0) exited kernel, duration=%lu ns\n", tid_x, duration);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
