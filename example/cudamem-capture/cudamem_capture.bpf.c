/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

SEC("uprobe")
int do_error_inject_patch(struct pt_regs *ctx)
{
	int rand = bpf_get_prandom_u32();
	if (rand % 2 == 0) {
		bpf_printk("bpf: Inject error. Target func will not exec.\n");
		bpf_override_return(ctx, -1);
		return 0;
	}
	bpf_printk("bpf: Continue.\n");
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
