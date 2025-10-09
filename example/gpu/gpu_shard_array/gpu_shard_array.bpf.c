#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, 4);
	__type(key, u32);
	__type(value, u64);
} counter SEC(".maps");

// 在被跟踪的 CUDA kernel 返回点累加一个固定 key 的计数
SEC("kretprobe/vectorAdd")
int cuda__retprobe()
{
	u32 key = 0;
	u64 *val = bpf_map_lookup_elem(&counter, &key);
	u64 newv = 1;
	if (val)
		newv = *val + 1;
	bpf_map_update_elem(&counter, &key, &newv, BPF_ANY);
	char msg[] = "gpu_update\\n";
	bpf_trace_printk(msg, sizeof(msg));
	return 0;
}

SEC("kprobe/vectorAdd")
int cuda__probe()
{
	u32 key = 0;
	u64 *val = bpf_map_lookup_elem(&counter, &key);
	u64 newv = 1;
	if (val)
		newv = *val + 1;
	bpf_map_update_elem(&counter, &key, &newv, BPF_ANY);
	char msg[] = "gpu_update_kprobe\\n";
	bpf_trace_printk(msg, sizeof(msg));
	return 0;
}

// (备用触发点已移除，避免 auto-attach 失败导致整体失败)

char LICENSE[] SEC("license") = "GPL";
