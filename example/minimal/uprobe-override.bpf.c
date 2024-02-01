/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

SEC("uprobe/_PyTime_GetSystemClock")
int _PyTime_GetSystemClock_bpf(struct pt_regs *ctx)
{
    u64 id = bpf_get_current_pid_tgid();
    u32 pid = id >> 32;
    bpf_printk("[_PyTime_GetSystemClock_bpf]Info: called :%lu", pid);

    u64 time = (PT_REGS_RC(ctx));

    bpf_printk("[_PyTime_GetSystemClock_bpf]Info: called :%lu & time is:%llu", pid, time);
    u64 t = 1706533301399118410; // Some time value to overwrite
    long ret = bpf_override_return(ctx, t);
    if (ret != 0)
    {
        bpf_printk("[_PyTime_GetSystemClock_bpf]Error: bpf_override_return failed");
        return 0;
    }

    // Print the process pid
    bpf_printk("[_PyTime_GetSystemClock_bpf]Info: Modified time for pid: %lu & time is:%llu", pid, t);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";