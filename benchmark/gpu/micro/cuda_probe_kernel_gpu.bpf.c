#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP 1502
#define BPF_MAP_TYPE_GPU_RINGBUF_MAP 1527
#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503
#define BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP 1504


// kernel-gpu shared array
struct {
	__uint(type, BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP);
	__uint(max_entries, 4);
	__type(key, u32);
	__type(value, u64);
	__uint(map_flags, BPF_F_MMAPABLE);
} kernel_gpu_shared_array SEC(".maps");

static const u64 (*bpf_get_globaltimer)(void) = (void *)502;


SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int cuda__retprobe_kernel_gpu_shared_array_update()
{
	u32 key = 0;
	u64 *val = bpf_map_lookup_elem(&kernel_gpu_shared_array, &key);
	u64 newv = 1;
	if (val)
		newv = *val + 1;
	bpf_map_update_elem(&kernel_gpu_shared_array, &key, &newv, BPF_ANY);
    return 0;
}

SEC("kretprobe/_Z9vectorAddPKfS0_Pf")
int cuda__retprobe_kernel_gpu_shared_array_lookup()
{
	u32 key = 0;
	u64 *val = bpf_map_lookup_elem(&kernel_gpu_shared_array, &key);
	if (val) {
		// Read value from GPU array
	}
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
