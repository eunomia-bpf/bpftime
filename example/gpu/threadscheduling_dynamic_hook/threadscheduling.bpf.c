#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_HASH_MAP 1501

struct thread_map {
	u64 sm_id;
	u64 warp_id;
	u64 lane_id;
	u64 block_x;
	u64 block_y;
	u64 block_z;
	u64 thread_x;
	u64 thread_y;
	u64 thread_z;
	u64 timestamp;
};

struct {
	__uint(type, BPF_MAP_TYPE_GPU_HASH_MAP);
	__uint(max_entries, 4096);
	__type(key, u32);
	__type(value, struct thread_map);
} thread_mapping SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_GPU_HASH_MAP);
	__uint(max_entries, 128);
	__type(key, u32);
	__type(value, u64);
} sm_histogram SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_GPU_HASH_MAP);
	__uint(max_entries, 1024);
	__type(key, u32);
	__type(value, u64);
} warp_histogram SEC(".maps");

static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;
static const u64 (*bpf_get_sm_id)(void) = (void *)509;
static const u64 (*bpf_get_warp_id)(void) = (void *)510;
static const u64 (*bpf_get_lane_id)(void) = (void *)511;

SEC("kprobe/_Z9vectorAddPKfS0_Pfi")
int cuda__probe_threadscheduling()
{
	u64 sm_id = bpf_get_sm_id();
	u64 warp_id = bpf_get_warp_id();
	u64 lane_id = bpf_get_lane_id();

	u64 block_x, block_y, block_z;
	u64 thread_x, thread_y, thread_z;
	bpf_get_block_idx(&block_x, &block_y, &block_z);
	bpf_get_thread_idx(&thread_x, &thread_y, &thread_z);

	u32 key = (u32)((block_x << 16) | (thread_x & 0xFFFF));

	struct thread_map info = {
		.sm_id = sm_id,
		.warp_id = warp_id,
		.lane_id = lane_id,
		.block_x = block_x,
		.block_y = block_y,
		.block_z = block_z,
		.thread_x = thread_x,
		.thread_y = thread_y,
		.thread_z = thread_z,
		.timestamp = bpf_get_globaltimer(),
	};
	bpf_map_update_elem(&thread_mapping, &key, &info, BPF_ANY);

	u32 sm_key = (u32)sm_id;
	u64 *sm_count = bpf_map_lookup_elem(&sm_histogram, &sm_key);
	if (sm_count) {
		__sync_fetch_and_add(sm_count, 1);
	} else {
		u64 initial = 1;
		bpf_map_update_elem(&sm_histogram, &sm_key, &initial, BPF_NOEXIST);
	}

	u32 warp_key = (u32)((sm_id << 16) | (warp_id & 0xFFFF));
	u64 *warp_count = bpf_map_lookup_elem(&warp_histogram, &warp_key);
	if (warp_count) {
		__sync_fetch_and_add(warp_count, 1);
	} else {
		u64 initial = 1;
		bpf_map_update_elem(&warp_histogram, &warp_key, &initial, BPF_NOEXIST);
	}

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
