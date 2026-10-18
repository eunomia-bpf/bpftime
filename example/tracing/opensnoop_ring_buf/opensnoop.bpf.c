// SPDX-License-Identifier: GPL-2.0
// Copyright (c) 2019 Facebook
// Copyright (c) 2020 Netflix
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include "opensnoop.h"

const volatile bool targ_failed = false;

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u32);
	__type(value, struct args_t);
} start SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 256 * 1024);
} rb SEC(".maps");

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
	// if (trace_allowed(tgid, pid)) {
	// since we can manually specify the attach process in userspace,
	// we don't need to check the process allowed here
	struct args_t args = {};
	args.fname = (const char *)ctx->args[1];
	args.flags = (int)ctx->args[2];
	bpf_map_update_elem(&start, &pid, &args, 0);
	// }
	return 0;
}

static __always_inline int trace_exit(struct trace_event_raw_sys_exit *ctx)
{
	struct event *event;
	struct args_t *ap;
	uintptr_t stack[3];
	int ret;
	u32 pid = bpf_get_current_pid_tgid();

	ap = bpf_map_lookup_elem(&start, &pid);
	if (!ap)
		return 0; /* missed entry */
	ret = ctx->ret;

	// if (targ_failed && ret >= 0)
	// 	goto cleanup; /* want failed only */
	event = bpf_ringbuf_reserve(&rb, sizeof(*event), 0);
	if (!event)
		return 0;

	/* event data */
	event->pid = bpf_get_current_pid_tgid() >> 32;
	event->uid = bpf_get_current_uid_gid();
	bpf_get_current_comm(&event->comm, sizeof(event->comm));
	bpf_probe_read_user_str(&event->fname, sizeof(event->fname), ap->fname);
	event->flags = ap->flags;
	event->ret = ret;
	bpf_get_stack(ctx, &stack, sizeof(stack), BPF_F_USER_STACK);

	/* Skip the first address that is usually the syscall it-self */
	event->callers[0] = stack[1];
	event->callers[1] = stack[2];

	/* emit event */
	bpf_ringbuf_submit(event, 0);
	return 0;
cleanup:
	bpf_map_delete_elem(&start, &pid);
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
