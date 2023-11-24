/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

SEC("tracepoint/syscalls/sys_enter_write")
int do_override_inject_syscall_sys_enter_write(
	struct trace_event_raw_sys_enter *ctx)
{
	int rand = bpf_get_prandom_u32();
	if (rand % 2 == 0) {
		bpf_printk("bpf: Inject error. sys_enter_write will failed.\n");
		bpf_override_return(ctx, -1);
		return 0;
	}
	bpf_printk("bpf: Continue.\n");
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
