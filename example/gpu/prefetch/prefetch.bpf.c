#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503
#define PREFETCH_MAX_CHUNKS_PER_THREAD 8
#define PREFETCH_MAX_PAGES 16

struct RunSeqConfig {
	int numBlocks;
	int blockSize;
	float *input;
	float *output;
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

static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;
static const u64 (*bpf_prefetch_l2)(const void *addr) = (void *)512;

SEC("uprobe/./prefetch_example:launch_run_seq_kernel")
int BPF_UPROBE(uprobe_func, struct RunSeqConfig *arg1)
{
	bpf_printk("Got RunSeqConfig\n");
	u32 key = 0;
	bpf_map_update_elem(&config_store, &key, arg1, BPF_ANY);
	return 0;
}

SEC("kprobe/_Z16seq_chunk_kernelPKfPfmmimi")
int cuda__prefetch_probe()
{
	u64 block_idx, block_dim, thread_idx;
	u64 y, z;

	u32 key = 0;
	struct RunSeqConfig *config = bpf_map_lookup_elem(&config_store, &key);
	if (config) {
		register float *input = config->input;
		register float *output = config->output;
		u64 N = config->N;
		u64 chunk_elems = config->chunk_elems;
		int chunks_per_thread = config->chunks_per_thread;
		int prefetch_pages = config->prefetch_pages;

		if (!input || !output || N == 0 || chunk_elems == 0 ||
		    chunks_per_thread <= 0 || prefetch_pages <= 0)
			return 0;

		if (chunks_per_thread > PREFETCH_MAX_CHUNKS_PER_THREAD)
			chunks_per_thread = PREFETCH_MAX_CHUNKS_PER_THREAD;
		if (prefetch_pages > PREFETCH_MAX_PAGES)
			prefetch_pages = PREFETCH_MAX_PAGES;

		bpf_get_block_idx(&block_idx, &y, &z);
		bpf_get_block_dim(&block_dim, &y, &z);
		bpf_get_thread_idx(&thread_idx, &y, &z);
		u64 tid = block_idx * block_dim + thread_idx;
		const size_t elems_per_page = 4096 / sizeof(float);
		for (int c = 0; c < PREFETCH_MAX_CHUNKS_PER_THREAD; ++c) {
			if (c >= chunks_per_thread)
				continue;
			size_t chunk_id = (size_t)tid * chunks_per_thread + c;
			size_t chunk_start = chunk_id * chunk_elems;

			if (chunk_start >= N)
				continue;

			for (int p = 0; p < PREFETCH_MAX_PAGES; ++p) {
				if (p >= prefetch_pages)
					continue;
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
