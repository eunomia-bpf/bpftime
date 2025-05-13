#ifndef __CUDA_NEURO_SURGEON_BPF_H
#define __CUDA_NEURO_SURGEON_BPF_H

#include <linux/types.h>

// Structure to track inference statistics
struct inference_stats {
	__u64 cpu_inference_count;
	__u64 gpu_inference_count;
	__u64 last_cpu_usage;
	__u64 last_gpu_usage;
	__u64 total_latency;
};

#endif /* __CUDA_NEURO_SURGEON_BPF_H */
