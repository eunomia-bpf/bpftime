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

