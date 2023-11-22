// SPDX-License-Identifier: GPL-2.0
// Copyright (c) 2019 Facebook
// Copyright (c) 2020 Netflix
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include "fs-filter-cache.h"

const volatile bool targ_failed = false;

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u32);
	__type(value, struct args_t);
} start SEC(".maps");

SEC("tracepoint/syscalls/sys_enter_open")
int tracepoint__syscalls__sys_enter_open(struct trace_event_raw_sys_enter *ctx)
{
	u64 id = bpf_get_current_pid_tgid();
	/* use kernel terminology here for tgid/pid: */
	u32 tgid = id >> 32;
	u32 pid = id;

	/* store arg info for later lookup */
	struct args_t args = {};
	args.fname = (const char *)ctx->args[0];
	args.flags = (int)ctx->args[1];
	bpf_map_update_elem(&start, &pid, &args, 0);
	bpf_printk("trace_enter sys_enter_open\n");
	return 0;
}

SEC("tracepoint/syscalls/sys_enter_openat")
int tracepoint__syscalls__sys_enter_openat(struct trace_event_raw_sys_enter *ctx)
{
	u64 id = bpf_get_current_pid_tgid();
	/* use kernel terminology here for tgid/pid: */
	u32 tgid = id >> 32;
	u32 pid = id;
	/* store arg info for later lookup */
	struct args_t args = {};
	args.fname = (const char *)ctx->args[1];
	args.flags = (int)ctx->args[2];
	bpf_map_update_elem(&start, &pid, &args, 0);
	bpf_printk("trace_enter sys_enter_openat\n");
	return 0;
}

SEC("tracepoint/syscalls/sys_enter_newfstatat")
int tracepoint__syscalls__sys_enter_newfstatat(struct trace_event_raw_sys_enter *ctx)
{
	bpf_printk("trace_enter sys_enter_newfstatat\n");
	return 0;
}

SEC("tracepoint/syscalls/sys_enter_statfs")
int tracepoint__syscalls__sys_enter_statfs(struct trace_event_raw_sys_enter *ctx)
{
	bpf_printk("trace_enter sys_enter_statfs\n");
	return 0;
}

SEC("tracepoint/syscalls/sys_enter_getdents64")
int tracepoint__syscalls__sys_enter_getdents64(struct trace_event_raw_sys_enter *ctx)
{
	bpf_printk("trace_enter sys_enter_getdents64\n");
	return 0;
}

static __always_inline int trace_exit(struct trace_event_raw_sys_exit *ctx)
{
	bpf_printk("trace_exit\n");
	return 0;
}

SEC("tracepoint/syscalls/sys_exit_open")
int tracepoint__syscalls__sys_exit_open(struct trace_event_raw_sys_exit *ctx)
{
	return trace_exit(ctx);
}

SEC("tracepoint/syscalls/sys_exit_openat")
int tracepoint__syscalls__sys_exit_openat(struct trace_event_raw_sys_exit *ctx)
{
	return trace_exit(ctx);
}

char LICENSE[] SEC("license") = "GPL";
