#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, u32);
    __type(value, u64);
} cuda_launch_calls_total SEC(".maps");

SEC("uprobe")
int count_cuda_launch(struct pt_regs *ctx)
{
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 zero = 0;
    u64 *count = bpf_map_lookup_elem(&cuda_launch_calls_total, &pid);
    if (!count) {
        bpf_map_update_elem(&cuda_launch_calls_total, &pid, &zero, BPF_NOEXIST);
        count = bpf_map_lookup_elem(&cuda_launch_calls_total, &pid);
        if (!count) {
            return 0;
        }
    }
    u64 updated = *count + 1;
    bpf_map_update_elem(&cuda_launch_calls_total, &pid, &updated, BPF_EXIST);
    bpf_printk("cudaLaunchKernel pid=%d count=%llu\n", pid, updated);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
