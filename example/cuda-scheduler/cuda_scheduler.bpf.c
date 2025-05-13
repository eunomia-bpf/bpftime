// 文件：gpu_fairshare_uprobe.bpf.c
#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

// 1) 存储每个 PID 上一次 entry 时间戳
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, u32);
	__type(value, u64);
} start_ts SEC(".maps");

// 2) 存储每个 PID 累积的执行时长（ns）
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, u32);
	__type(value, u64);
} total_time_ns SEC(".maps");

// 3) 存储每个 PID 的调用次数
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, u32);
	__type(value, u64);
} call_count SEC(".maps");
// 不schedule的kernel
// uprobe：在 matMulTiled 入口处记录时间戳，并给调用次数+1
SEC("kprobe/_Z11matMulTiledPKfS0_Pf")
int uprobe_matMulTiled(struct pt_regs *ctx)
{
	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u64 ts = bpf_ktime_get_ns();

	// 记录入口时间
	bpf_map_update_elem(&start_ts, &pid, &ts, BPF_ANY);

	// 累加调用次数
	u64 one = 1;
    u64 *count = bpf_map_lookup_elem(&call_count, &pid);
    if (count) {
        *count += 1;
    } else {
        bpf_map_update_elem(&call_count, &pid, &one, BPF_ANY);
    }

	return 0;
}

// uretprobe：在 matMulTiled 返回时计算执行时长，累加到 total_time_ns，并清除
// start_ts
SEC("kretprobe/_Z11matMulTiledPKfS0_Pf")
int uretprobe_matMulTiled(struct pt_regs *ctx)
{
	u32 pid = bpf_get_current_pid_tgid() >> 32;

	// 查上次 entry 的时间戳
	u64 *tsp = bpf_map_lookup_elem(&start_ts, &pid);
	if (tsp) {
		u64 delta = bpf_ktime_get_ns() - *tsp;
		bpf_map_delete_elem(&start_ts, &pid);

		// 累加到 total_time_ns
		u64 *total = bpf_map_lookup_elem(&total_time_ns, &pid);
		if (total) {
			*total += delta;
			bpf_map_update_elem(&total_time_ns, &pid, total,
					    BPF_EXIST);
		} else {
			bpf_map_update_elem(&total_time_ns, &pid, &delta,
					    BPF_NOEXIST);
		}
	}

	return 0;
}

char LICENSE[] SEC("license") = "GPL";