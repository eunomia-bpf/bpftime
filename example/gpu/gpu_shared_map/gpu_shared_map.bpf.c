#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_HASH_MAP 1501
#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503

static const u64 (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64 (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;

// Device-side direct write path is enabled for GPU_ARRAY_MAP in the trampoline;
// updates are memcpy overwrites (non-atomic), visible to host after system
// fence.

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, u64);
} counter SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_GPU_HASH_MAP);
	__uint(max_entries, 2);
	__type(key, u32);
	__type(value, u64);
} counter_per_thread SEC(".maps");

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
	
	u64 x, y, z;
	bpf_get_thread_idx(&x, &y, &z);
	u64 bx, by, bz;
	bpf_get_block_idx(&bx, &by, &bz);
	u64 idx = x+y+z+bx+by+bz;
	if(idx == 0){
		char msg[] = "gpu_update\n";
		bpf_trace_printk(msg, sizeof(msg));
		u32 pid = (bpf_get_current_pid_tgid() >> 32);
		// update (Host-to-Device memcpy in slow path)
		bpf_map_update_elem(&counter_per_thread, &pid, &newv, (u64)BPF_ANY);
	}
	return 0;
}

SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__probe()
{
	u32 key = 0;
	u64 *val = bpf_map_lookup_elem(&counter, &key);

	u64 x, y, z;
	bpf_get_thread_idx(&x, &y, &z);
	u64 bx, by, bz;
	bpf_get_block_idx(&bx, &by, &bz);
	u64 idx = x+y+z+bx+by+bz;
	if(idx == 0){
		char msg[] = "gpu_update_kprobe\n";
		bpf_trace_printk(msg, sizeof(msg));
	}
	return 0;
}

// (Backup trigger removed to avoid auto-attach failure from causing overall
// failure)

char LICENSE[] SEC("license") = "GPL";
