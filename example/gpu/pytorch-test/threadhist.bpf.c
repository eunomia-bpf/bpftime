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

SEC("kretprobe/_ZN2at6native20bitonicSortKVInPlaceILin2ELin1ELi16ELi16EilNS0_4LTOpIiLb1EEEjEEvNS_4cuda6detail10TensorInfoIT3_T6_EES8_S8_S8_NS6_IT4_S8_EES8_T5_")
int cuda__retprobe()
{
	u32 key = 0;
	u64 *cnt = bpf_map_lookup_elem(&call_count, &key);
	if (cnt)
		*cnt += 1;

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
