#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP 1504

struct bb_reg_snapshot {
	__u64 hits;
	__u64 last_r2;
	__u64 last_rd8;
};

struct {
	__uint(type, BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP);
	__uint(max_entries, 1);
	__type(key, __u32);
	__type(value, struct bb_reg_snapshot);
} bb_regs SEC(".maps");

static const u64 (*bpf_get_ptx_reg)(u64 idx, u64 ctx, u64 _1, u64 _2, u64 _3) =
	(void *)512;

/*
 * Capture %r2 and %rd8 at BB4 in bb_reg_kernel.
 * The order of bpf_get_ptx_reg(idx, ...) follows the suffix order:
 *   __r2__rd8 -> idx 0 is r2, idx 1 is rd8.
 */
SEC("kprobe/bb_reg_kernel__BB4__r2__rd8")
int cuda__bb_register_capture(void *ctx)
{
	u32 key = 0;
	struct bb_reg_snapshot *snap;
	struct bb_reg_snapshot next = {};

	snap = bpf_map_lookup_elem(&bb_regs, &key);
	if (snap)
		next.hits = snap->hits + 1;
	else
		next.hits = 1;
	next.last_r2 = bpf_get_ptx_reg(0, (u64)ctx, 0, 0, 0);
	next.last_rd8 = bpf_get_ptx_reg(1, (u64)ctx, 0, 0, 0);
	bpf_map_update_elem(&bb_regs, &key, &next, 0);

	bpf_printk("bb_reg_kernel BB4: hit=%llu r2=0x%llx rd8=0x%llx\n",
		   next.hits, next.last_r2, next.last_rd8);
	return 0;
}

char LICENSE[] SEC("license") = "Dual BSD/GPL";
