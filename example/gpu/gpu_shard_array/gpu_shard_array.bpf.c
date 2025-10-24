#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503
// Device-side direct write path is enabled for GPU_ARRAY_MAP in the trampoline;
// updates are memcpy overwrites (non-atomic), visible to host after system
// fence.

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, 4);
	__type(key, u32);
	__type(value, u64);
} counter SEC(".maps");

// At the CUDA kernel return point, update the counter for a fixed key.
// NOTE: Non-atomic overwrite semantics; last-writer-wins. For accurate sums,
// aggregate before write or shard keys to avoid contention.
SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int cuda__retprobe()
{
	u32 key = 0;
	u64 *val = bpf_map_lookup_elem(&counter, &key);
	// Read, add, and write back as an overwrite (device-side memcpy in
	// trampoline)
	u64 newv = 1;
	if (val)
		newv = *val + 1;
	bpf_map_update_elem(&counter, &key, &newv, (u64)BPF_ANY);
	char msg[] = "gpu_update\\n";
	bpf_trace_printk(msg, sizeof(msg));
	return 0;
}

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe()
{
	u32 key = 0;
	u64 *val = bpf_map_lookup_elem(&counter, &key);
	char msg[] = "gpu_update_kprobe\\n";
	bpf_trace_printk(msg, sizeof(msg));
	return 0;
}

// (Backup trigger removed to avoid auto-attach failure from causing overall
// failure)

char LICENSE[] SEC("license") = "GPL";
