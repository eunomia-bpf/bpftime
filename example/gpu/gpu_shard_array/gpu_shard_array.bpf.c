#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503
// High-32 bits are reserved for bpftime-specific map ops.
// When BPFTIME_UPDATE_OP_ADD is set in the high bits of 'flags',
// map_update_elem requests a host-side fetch_add:
//   - 'value' is interpreted as u64 delta
//   - backend does RMW: load->add->store on the target slot
#define BPFTIME_UPDATE_OP_ADD (1ULL << 32)

struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, 4);
	__type(key, u32);
	__type(value, u64);
} counter SEC(".maps");

// At the CUDA kernel return point, increment the counter for a fixed key.
// NOTE: This uses host-side fetch_add (not GPU-side atomic) and relies on
// external serialization when multiple writers might contend.
SEC("kretprobe/vectorAdd")
int cuda__retprobe()
{
	u32 key = 0;
	u64 *val = bpf_map_lookup_elem(&counter, &key);
	// Use host-side atomic ADD semantics: pass delta=1 and set high-bit op
	// flag
	u64 delta = 1;
	bpf_map_update_elem(&counter, &key, &delta,
			    (u64)BPF_ANY | BPFTIME_UPDATE_OP_ADD);
	char msg[] = "gpu_update\\n";
	bpf_trace_printk(msg, sizeof(msg));
	return 0;
}

SEC("kprobe/vectorAdd")
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
