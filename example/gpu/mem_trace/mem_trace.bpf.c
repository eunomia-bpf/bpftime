/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2025, eunomia-bpf org
 * All rights reserved.
 */
#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define BPF_MAP_TYPE_GPU_ARRAY_MAP 1503


struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_MAP);
	__uint(max_entries, 4);
	__type(key, u32);
	__type(value, u64);
} counter SEC(".maps");

SEC("kprobe/__memcapture")
int cuda__trace_cuda_kernel(struct pt_regs *ctx)
{
	// bpf_printk("mem_trace called");
	u32 key = 0;
	u64 *val = bpf_map_lookup_elem(&counter, &key);
	if (val)
		*val = *val + 1;
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
