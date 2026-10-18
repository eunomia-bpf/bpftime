#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} partition_num_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} partition_index_map SEC(".maps");

static const void (*ebpf_puts)(const char *) = (void *)501;
static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;
static const u64 (*bpf_cuda_exit)() = (void *)507;
static const u64 (*bpf_get_grid_dim)(u64 *x, u64 *y, u64 *z) = (void *)508;

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe()
{
	u32 key = 0;
	u64 x, y, z;
	bpf_get_block_idx(&x, &y, &z);
	u64 gx, gy, gz;
	bpf_get_grid_dim(&gx, &gy, &gz);
	
	u64 block_num = gx * gy * gz;
	u64 block_id = z * gy * gx + y * gx + x;

	int partition_num = 0;
	int partition_index = 0;
	
	u64 *partition_num_ptr = bpf_map_lookup_elem(&partition_num_map, &key);
	if (partition_num_ptr) {
		partition_num = *partition_num_ptr;
		bpf_printk("partition_num is %u for block %lu\n", partition_num, block_id);
	} else {
		bpf_printk("partition_num_ptr is null for block %lu\n", block_id);
		return 0;
	}

	u64 *partition_idx_ptr = bpf_map_lookup_elem(&partition_index_map, &key);
	if (partition_idx_ptr) {
		partition_index = *partition_idx_ptr;
		bpf_printk("partition_index is %u for block %lu\n", partition_index, block_id);
	} else {
		bpf_printk("partition_idx_ptr is null for block %lu\n", block_id);
		return 0;
	}

	u64 L = (block_num * partition_index) / partition_num;
	u64 H = (block_num * (partition_index + 1)) / partition_num;

	if (block_id < L || block_id >= H)
	{
		bpf_printk("Exited _Z9vectorAddPKfS0_Pf block_id=%lu, L=%lu, H=%lu\n", block_id, L, H);
		bpf_cuda_exit();
		// never reach here
		return 0;
	}

	bpf_printk("Enter _Z9vectorAddPKfS0_Pf block_id=%lu, L=%lu, H=%lu\n", block_id, L, H);

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
