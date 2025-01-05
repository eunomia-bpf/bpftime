#include <linux/sched.h>
#include <uapi/linux/bpf.h>

#define WINDOW_SEC     1                // 窗口大小（秒）
#define MAX_KERNELS   (WINDOW_SEC*100)  // 一个粗略上限，用于初始化

BPF_HASH(launch_count, u32, u64);        // PID -> 本窗口提交次数
BPF_HASH(window_start, u32, u64);        // PID -> 窗口起始时间(ns)
BPF_HASH(active_pids, u32, u8);          // PID -> 标记为活跃

SEC("kprobe/nvidia_ioctl")
int kprobe__nvidia_ioctl(struct pt_regs *ctx,
                         unsigned int cmd,
                         unsigned long arg)
{
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 now = bpf_ktime_get_ns();
    u64 ns_window = WINDOW_SEC * 1000000000ULL;

    // 标记为活跃
    u8 one = 1;
    active_pids.update(&pid, &one);

    // 初始化或重置窗口
    u64 *start = window_start.lookup(&pid);
    if (!start || now - *start > ns_window) {
        window_start.update(&pid, &now);
        u64 zero = 0;
        launch_count.update(&pid, &zero);
        start = window_start.lookup(&pid);
    }
    if (!start) return 0;

    // 增量计数
    u64 *cnt = launch_count.lookup_or_init(&pid, &(u64){0});
    (*cnt)++;
    launch_count.update(&pid, cnt);

    return 0;
}

char LICENSE[] SEC("license") = "GPL";