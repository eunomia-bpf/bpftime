/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2025, eunomia-bpf org
 * All rights reserved.
 */
#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

SEC("kprobe/__memcapture")
int cuda__trace_cuda_kernel(struct pt_regs *ctx)
{
	bpf_printk("mem_trace called");
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
