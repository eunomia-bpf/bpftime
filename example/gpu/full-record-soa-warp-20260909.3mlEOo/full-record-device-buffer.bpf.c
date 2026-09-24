#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503
#define FULL_RECORD_VALUE_U64 u64
#include "full_record_device_buffer_value.h"

static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;
static const u64 (*bpf_get_grid_dim)(u64 *x, u64 *y, u64 *z) = (void *)508;

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, FRDB_NUM_BANKS);
	__type(key, u32);
	__type(value, FRDB_VALUE_TYPE);
} arena SEC(".maps");

SEC("kretprobe/_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli")
int cuda__retprobe(void)
{
	struct frdb_record record;
	u64 block_x, block_y, block_z;
	u64 thread_x, thread_y, thread_z;
	u64 block_dim_x, block_dim_y, block_dim_z;
	u64 grid_x, grid_y, grid_z;
	u64 width, height, thread_id;
	u64 block_linear, lane_linear;
	u64 bank, slot, counter;
	frdb_value *value;
	u32 key;

	bpf_get_block_idx(&block_x, &block_y, &block_z);
	bpf_get_thread_idx(&thread_x, &thread_y, &thread_z);
	bpf_get_block_dim(&block_dim_x, &block_dim_y, &block_dim_z);
	bpf_get_grid_dim(&grid_x, &grid_y, &grid_z);

	record.block_x = block_x;
	record.block_y = block_y;
	record.block_z = block_z;
	record.thread_x = thread_x;
	record.thread_y = thread_y;
	record.thread_z = thread_z;
	record.block_dim_x = block_dim_x;
	record.block_dim_y = block_dim_y;
	record.block_dim_z = block_dim_z;
	record.timestamp = bpf_get_globaltimer();

#ifdef FRDB_WARP_CONTIGUOUS_INDEX
	/* Opt-in storage bijection for the fixed pp512 rope geometry
	 * (grid 2048x1x1, block 1x256x1): standard
	 * block-major/thread-major order, so adjacent physical warp
	 * lanes map to adjacent storage slots. Computed in u64 before
	 * any bank/slot indexing; out-of-range and overflow reporting
	 * are unchanged. */
	block_linear = ((block_z * grid_y) + block_y) * grid_x + block_x;
	lane_linear = ((thread_z * block_dim_y) + thread_y) * block_dim_x
			+ thread_x;
	thread_id = block_linear * (block_dim_x * block_dim_y
					* block_dim_z)
		   + lane_linear;
#else
	/* Existing per-thread linear coordinate (getGlobalThreadId),
	 * computed in u64 before any bank/slot indexing. */
	width = grid_x * block_dim_x;
	height = grid_y * block_dim_y;
	thread_id = (block_z * block_dim_z + thread_z) * width * height
		   + (block_y * block_dim_y + thread_y) * width
		   + (block_x * block_dim_x + thread_x);
#endif

	if (thread_id >= FRDB_TOTAL_SLOTS) {
		/* Outside the supported 524288-slot geometry: count it in the
		 * bank-0 sink and do not index the record storage. */
		key = 0;
		value = bpf_map_lookup_elem(&arena, &key);
		if (value)
			value->total_out_of_range += 1;
		return 0;
	}

	bank = thread_id / FRDB_SLOTS_PER_BANK;
	slot = thread_id % FRDB_SLOTS_PER_BANK;
	key = (u32)bank;
	value = bpf_map_lookup_elem(&arena, &key);
	if (!value)
		return 0;
	counter = value->slot_counters[slot];
	if (counter >= FRDB_RECORDS_PER_SLOT) {
		value->total_overflow += 1;
		return 0;
	}
	#if defined(FRDB_AOSOA_LAYOUT)
	{
		/* 32-slot grouped SoA (AoSoA): for record counter and slot,
		 * fields[counter][slot / 32][f][slot % 32] is u64 index
		 * (((counter * (16384 / 32) + slot / 32) * 10 + f) * 32 +
		 * slot % 32); the 32 lanes of one group are 8 bytes apart
		 * within each field, and a slot's ten fields stay inside
		 * 2560 contiguous bytes at this record index. */
		const u64 group = slot / FRDB_GROUP_SLOTS;
		const u64 lane = slot % FRDB_GROUP_SLOTS;

		value->fields[counter][group][0][lane] = record.block_x;
		value->fields[counter][group][1][lane] = record.block_y;
		value->fields[counter][group][2][lane] = record.block_z;
		value->fields[counter][group][3][lane] = record.thread_x;
		value->fields[counter][group][4][lane] = record.thread_y;
		value->fields[counter][group][5][lane] = record.thread_z;
		value->fields[counter][group][6][lane] = record.block_dim_x;
		value->fields[counter][group][7][lane] = record.block_dim_y;
		value->fields[counter][group][8][lane] = record.block_dim_z;
		value->fields[counter][group][9][lane] = record.timestamp;
	}
	#elif defined(FRDB_SOA_LAYOUT)
	{
		/* Field-major SoA: one 8-byte store per field plane; for a
		 * fixed record index, adjacent lanes are 8 bytes apart. */
		const u64 plane_index = counter * FRDB_SLOTS_PER_BANK + slot;

		value->block_x[plane_index] = record.block_x;
		value->block_y[plane_index] = record.block_y;
		value->block_z[plane_index] = record.block_z;
		value->thread_x[plane_index] = record.thread_x;
		value->thread_y[plane_index] = record.thread_y;
		value->thread_z[plane_index] = record.thread_z;
		value->block_dim_x[plane_index] = record.block_dim_x;
		value->block_dim_y[plane_index] = record.block_dim_y;
		value->block_dim_z[plane_index] = record.block_dim_z;
		value->timestamp[plane_index] = record.timestamp;
	}
	#else
	value->records[counter * FRDB_SLOTS_PER_BANK + slot] = record;
	#endif
	value->slot_counters[slot] = counter + 1;
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
