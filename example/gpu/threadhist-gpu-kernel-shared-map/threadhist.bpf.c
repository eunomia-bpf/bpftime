#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP 1504

struct {
	__uint(type, BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP);
	__uint(max_entries, 2);
	__type(key, u32);
	__type(value, u64);
	__uint(map_flags, BPF_F_MMAPABLE);
} call_count SEC(".maps");

static const void (*ebpf_puts)(const char *) = (void *)501;
static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

// Run by bpftime
SEC("kretprobe/cuda__Z9vectorAddPKfS0_Pf")
int cuda__retprobe()
{
	u32 key = 0;
	u64 *cnt = bpf_map_lookup_elem(&call_count, &key);
	if (cnt)
		__atomic_add_fetch(cnt, 1, __ATOMIC_SEQ_CST);

	return 0;
}

// Run by kernel
SEC("uretprobe/./vec_add:_Z16cudaLaunchKernelIcE9cudaErrorPT_4dim3S3_PPvmP11CUstream_st")
int launch_kernel_ret_probe()
{
	bpf_printk("triggered");
	u32 key = 1;
	u64 *cnt = bpf_map_lookup_elem(&call_count, &key);
	if (cnt)
		__atomic_add_fetch(cnt, 1, __ATOMIC_SEQ_CST);

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
