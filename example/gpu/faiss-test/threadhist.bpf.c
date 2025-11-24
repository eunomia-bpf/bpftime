#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP 1502

struct {
	__uint(type, BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} call_count SEC(".maps");

static const void (*ebpf_puts)(const char *) = (void *)501;
static const u64 (*bpf_get_globaltimer)(void) = (void *)502;
static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

SEC("kretprobe/_ZN5faiss3gpu14l2NormRowMajorIf6float4Li8ELb1EEEvNS0_6TensorIT0_Li2ELb1ElNS0_6traits16DefaultPtrTraitsEEENS3_IfLi1ELb1ElS6_EE")
int cuda__retprobe()
{
	u32 key = 0;
	u64 *cnt = bpf_map_lookup_elem(&call_count, &key);
	if (cnt)
		*cnt += 1;

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
