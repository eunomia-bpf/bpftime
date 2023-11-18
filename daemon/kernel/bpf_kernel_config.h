/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef BPFTIME_KERNEL_CONFIG_H
#define BPFTIME_KERNEL_CONFIG_H

#include <vmlinux.h>

#define PATH_LENTH 255

// filter bpf program pid
const volatile int target_pid = 0;
// current bpf program pid, avoid breaking current process
const volatile int current_pid = 0;
// enable modify bpf program
const volatile bool enable_replace_prog = 0;
// enable modify uprobe
const volatile bool enable_replace_uprobe = 0;
const char new_uprobe_path[PATH_LENTH] = "\0";

const volatile int uprobe_perf_type = 0;
const volatile int kprobe_perf_type = 0;

const volatile bool submit_bpf_events = 0;

static __always_inline bool filter_target(void)
{
	u64 pid = bpf_get_current_pid_tgid() & 0xffffffff;
	if (target_pid && pid != target_pid) {
		// filter target pid
		return false;
	}
	if (current_pid && pid == current_pid) {
		// avoid breaking current process
		return false;
	}
	return true;
}

#endif // BPFTIME_KERNEL_CONFIG_H
