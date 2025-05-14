#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

// 记录入口时间戳
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, u32);
    __type(value, u64);
} start_ts SEC(".maps");

// 累积执行时间(ns)
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, u32);
    __type(value, u64);
} total_time_ns SEC(".maps");

// 调用次数
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, u32);
    __type(value, u64);
} call_count SEC(".maps");

// 公平抢占：当前被允许运行的 PID
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u32);
} run_pid_map SEC(".maps");

// GPU kernel 入口 kprobe
SEC("kprobe/_Z10bfs_kernelPKiS0_S0_iPiS1_S1_i")
int kprobe_bfs_entry__cuda(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 ts = bpf_ktime_get_ns();
    bpf_map_update_elem(&start_ts, &pid, &ts, BPF_ANY);

    u64 one = 1;
    u64 *cnt = bpf_map_lookup_elem(&call_count, &pid);
    if (cnt) {
        *cnt += 1;
        bpf_map_update_elem(&call_count, &pid, cnt, BPF_EXIST);
    } else {
        bpf_map_update_elem(&call_count, &pid, &one, BPF_NOEXIST);
    }
    return 0;
}

// GPU kernel 退出 kretprobe
SEC("kretprobe/_Z10bfs_kernelPKiS0_S0_iPiS1_S1_i")
int kretprobe_bfs_exit__cuda(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 *tsp = bpf_map_lookup_elem(&start_ts, &pid);
    if (tsp) {
        u64 delta = bpf_ktime_get_ns() - *tsp;
        bpf_map_delete_elem(&start_ts, &pid);
        u64 *total = bpf_map_lookup_elem(&total_time_ns, &pid);
        if (total) {
            *total += delta;
            bpf_map_update_elem(&total_time_ns, &pid, total, BPF_EXIST);
        } else {
            bpf_map_update_elem(&total_time_ns, &pid, &delta, BPF_NOEXIST);
        }
    }

    // 自旋等待被调度
    u32 key = 0;
    u32 *allowed;
    #pragma unroll 1
    while (1) {
        allowed = bpf_map_lookup_elem(&run_pid_map, &key);
        if (allowed && *allowed == pid) {
            u32 zero = 0;
            bpf_map_update_elem(&run_pid_map, &key, &zero, BPF_ANY);
            break;
        }
    }
    return 0;
}

char LICENSE[] SEC("license") = "GPL";