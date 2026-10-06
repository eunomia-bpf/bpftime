#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>
#include "perthreadRuntimeDist.h"


struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, sizeof(__u32));
} events SEC(".maps");

// 用于记录 start time
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, u32);
    __type(value, u64);
} start SEC(".maps");

char LICENSE[] SEC("license") = "Dual BSD/GPL";

// EXT helper index — 必须存在且可见
static const u64 (*bpf_get_globaltimer)(void) = (void *)502;

// GPU kernel entry
SEC("kprobe/cudaLaunchKernel")
int cuda__kernel_entry(struct pt_regs *ctx)
{
    u32 tid = bpf_get_current_pid_tgid();

    u64 start_cycles = bpf_get_globaltimer(); // bpf_ktime_get_ns()?

    bpf_map_update_elem(&start, &tid, &start_cycles, BPF_ANY);

    return 0;
}

SEC("kretprobe/cudaLaunchKernel")
int cuda__kernel_exit(struct pt_regs *ctx)
{
    u32 tid = bpf_get_current_pid_tgid();
    u64 *start_cycles = bpf_map_lookup_elem(&start, &tid);

    if (!start_cycles)
        return 0;

    u64 end_cycles = bpf_get_globaltimer();

    struct event_t evt = {
        .tid = tid,
        .cycles = end_cycles - *start_cycles,
    };

    bpf_perf_event_output(ctx, &events,
                          BPF_F_CURRENT_CPU,
                          &evt,
                          sizeof(evt));

    bpf_map_delete_elem(&start, &tid);
    return 0;
}
