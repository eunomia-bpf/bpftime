#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP 1502
#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503
struct RunSeqConfig {
	int numBlocks;
	int blockSize;
	u32 *input;
	u32 *output;
	unsigned long N;
	unsigned long chunk_elems;
	int chunks_per_thread;
	unsigned long stride_elems;
	int prefetch_pages;
};

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct RunSeqConfig);
} config_store SEC(".maps");

static const void (*ebpf_puts)(const char *) = (void *)501;
static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;
static const u64 (*bpf_prefetch_l2)(void *addr) = (void *)509;

SEC("uprobe/./prefetch_example:launch_run_seq_kernel")
int BPF_UPROBE(uprobe_func, struct RunSeqConfig *arg1)
{
	bpf_printk("Got RunSeqconfig\n");
	u32 key = 0;
	bpf_map_update_elem(&config_store, &key, arg1, BPF_ANY);
	return 0;
}

SEC("kprobe/_Z16seq_chunk_kernelPKfPfmmimi")
int cuda__retprobe()
{
	u64 block_idx, block_dim, thread_idx;
	u64 y, z;

	u32 key = 0;
	struct RunSeqConfig *config = bpf_map_lookup_elem(&config_store, &key);
	if (config) {
		register u32 *input = config->input;
		register u32 *output = config->output;
#define N 4503490560ul
#define chunk_elems 69632
#define chunks_per_thread 1
#define stride_elems 1024
#define prefetch_pages 4
		bpf_get_block_idx(&block_idx, &y, &z);
		bpf_get_block_dim(&block_dim, &y, &z);
		bpf_get_thread_idx(&thread_idx, &y, &z);
		u64 tid = block_idx * block_dim + thread_idx;
		const size_t elems_per_page = 4096 / sizeof(float);
#pragma unroll
		for (int c = 0; c < chunks_per_thread; ++c) {
			size_t chunk_id = (size_t)tid * chunks_per_thread + c;
			size_t chunk_start = chunk_id * chunk_elems;

			if (chunk_start >= N)
				break;

#pragma unroll
			for (int p = 0; p < prefetch_pages; ++p) {
				size_t pf_addr =
					chunk_start + p * elems_per_page;
				if (pf_addr < N) {
					bpf_prefetch_l2(&input[pf_addr]);
					bpf_prefetch_l2(&output[pf_addr]);
				}
			}
		}
	}
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
