#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "cuda_neuro_surgeon.bpf.h"

// Map to track CPU/GPU usage
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, u32); // pid
	__type(value, u64); // timestamp of last CPU/GPU usage
} cpu_gpu_usage SEC(".maps");

// Map to store scheduling decisions
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, u32); // pid
	__type(value, u32); // 0 for CPU, 1 for GPU
} scheduler_decisions SEC(".maps");

// Map to track inference statistics
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, u32); // pid
	__type(value, struct inference_stats);
} inference_stats SEC(".maps");

char LICENSE[] SEC("license") = "GPL";

// Helper function to get current timestamp
static inline u64 get_timestamp(void)
{
	return bpf_ktime_get_ns();
}

// CPU inference handler
SEC("tp/syscalls/sys_enter_write")
int handle_cpu_inference(struct trace_event_raw_sys_enter *ctx)
{
	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u64 ts = get_timestamp();

	// Update CPU usage timestamp
	bpf_map_update_elem(&cpu_gpu_usage, &pid, &ts, BPF_ANY);

	// Check if we should use CPU
	u32 *decision = bpf_map_lookup_elem(&scheduler_decisions, &pid);
	if (!decision || *decision == 0) {
		// CPU is chosen, proceed with inference
		struct inference_stats *stats =
			bpf_map_lookup_elem(&inference_stats, &pid);
		if (stats) {
			stats->cpu_inference_count++;
			stats->last_cpu_usage = ts;
		}
		return 0;
	}

	// GPU was chosen, sleep
	bpf_trace_printk("PID %d: CPU inference skipped, GPU chosen\n", pid);
	return 0;
}

// GPU inference handler
SEC("uprobe/libcuda.so:cudaLaunchKernel")
int handle_gpu_inference(struct pt_regs *ctx)
{
	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u64 ts = get_timestamp();

	// Update GPU usage timestamp
	bpf_map_update_elem(&cpu_gpu_usage, &pid, &ts, BPF_ANY);

	// Check if we should use GPU
	u32 *decision = bpf_map_lookup_elem(&scheduler_decisions, &pid);
	if (!decision || *decision == 1) {
		// GPU is chosen, proceed with inference
		struct inference_stats *stats =
			bpf_map_lookup_elem(&inference_stats, &pid);
		if (stats) {
			stats->gpu_inference_count++;
			stats->last_gpu_usage = ts;
		}
		return 0;
	}

	// CPU was chosen, sleep
	bpf_trace_printk("PID %d: GPU inference skipped, CPU chosen\n", pid);
	return 0;
}

// Scheduler function to make decisions
SEC("tp/syscalls/sys_enter_sched_yield")
int handle_scheduler(struct trace_event_raw_sys_enter *ctx)
{
	u32 pid = bpf_get_current_pid_tgid() >> 32;

	// Get current usage timestamps
	u64 *cpu_ts = bpf_map_lookup_elem(&cpu_gpu_usage, &pid);
	u64 current_ts = get_timestamp();

	if (!cpu_ts) {
		// First time, default to CPU
		u32 decision = 0;
		bpf_map_update_elem(&scheduler_decisions, &pid, &decision,
				    BPF_ANY);
		return 0;
	}

	// Simple round-robin scheduling
	u32 *current_decision = bpf_map_lookup_elem(&scheduler_decisions, &pid);
	if (current_decision) {
		u32 new_decision = (*current_decision + 1) % 2;
		bpf_map_update_elem(&scheduler_decisions, &pid, &new_decision,
				    BPF_ANY);

		bpf_trace_printk("PID %d: Scheduling decision changed to %s\n",
				 pid, new_decision ? "GPU" : "CPU");
	}

	return 0;
}
