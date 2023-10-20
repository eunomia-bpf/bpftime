// SPDX-License-Identifier: GPL-2.0
// Copyright (c) 2019 Facebook
// Copyright (c) 2020 Netflix
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "bpf-mocker-event.h"
#include "bpf-utils.h"

struct open_args_t {
	const char *fname;
	int flags;
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u32);
	__type(value, struct open_args_t);
} open_param_start SEC(".maps");

SEC("tracepoint/syscalls/sys_enter_open")
int tracepoint__syscalls__sys_enter_open(struct trace_event_raw_sys_enter *ctx)
{
	if (!filter_target()) {
		return 0;
	}
	/* store arg info for later lookup */
	struct open_args_t args = {};
	args.fname = (const char *)ctx->args[0];
	args.flags = (int)ctx->args[1];

	u32 pid = bpf_get_current_pid_tgid() >> 32;
	bpf_map_update_elem(&open_param_start, &pid, &args, 0);
	return 0;
}

SEC("tracepoint/syscalls/sys_enter_openat")
int tracepoint__syscalls__sys_enter_openat(struct trace_event_raw_sys_enter *ctx)
{
	if (!filter_target()) {
		return 0;
	}
	/* store arg info for later lookup */
	struct open_args_t args = {};
	args.fname = (const char *)ctx->args[1];
	args.flags = (int)ctx->args[2];

	u32 pid = bpf_get_current_pid_tgid() >> 32;
	bpf_map_update_elem(&open_param_start, &pid, &args, 0);
	return 0;
}

static __always_inline int trace_exit_open(struct trace_event_raw_sys_exit *ctx)
{
	struct event *event = NULL;
	struct open_args_t *ap = NULL;
	u32 pid = 0;

	if (!filter_target()) {
		return 0;
	}
	pid = bpf_get_current_pid_tgid() >> 32;
	ap = bpf_map_lookup_elem(&open_param_start, &pid);
	if (!ap)
		return 0; /* missed entry */

	/* event data */
	event = fill_basic_event_info();
	if (!event) {
		return 0;
	}

	event->type = SYS_OPEN;
	bpf_probe_read_user_str(&event->open_data.fname,
				sizeof(event->open_data.fname), ap->fname);
	event->open_data.flags = ap->flags;
	event->open_data.ret = ctx->ret;

	/* emit event */
	bpf_ringbuf_submit(event, 0);
cleanup:
	bpf_map_delete_elem(&open_param_start, &pid);
	return 0;
}

SEC("tracepoint/syscalls/sys_exit_open")
int tracepoint__syscalls__sys_exit_open(struct trace_event_raw_sys_exit *ctx)
{
	return trace_exit_open(ctx);
}

SEC("tracepoint/syscalls/sys_exit_openat")
int tracepoint__syscalls__sys_exit_openat(struct trace_event_raw_sys_exit *ctx)
{
	return trace_exit_open(ctx);
}

struct bpf_args_t {
	enum bpf_cmd cmd;
	void *attr;
	u32 attr_size;
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u32);
	__type(value, struct bpf_args_t);
} bpf_param_start SEC(".maps");

static bool should_modify_program(union bpf_attr *new_attr)
{
	if (new_attr->insn_cnt < 2 || disable_modify) {
		return false;
	}
	// Only target BPF_PROG_TYPE_KPROBE and BPF_PROG_TYPE_TRACEPOINT
	if (new_attr->prog_type != BPF_PROG_TYPE_KPROBE &&
	    new_attr->prog_type != BPF_PROG_TYPE_TRACEPOINT) {
		return false;
	}
	return true;
}

static int process_bpf_prog_load_events(union bpf_attr *attr)
{
	unsigned int insn_cnt;
	void *insns;
	struct event *event = NULL;
	union bpf_attr new_attr = { 0 };

	/* event data */
	event = fill_basic_event_info();
	if (!event) {
		return 0;
	}
	event->type = BPF_PROG_LOAD_EVENT;
	bpf_probe_read_user(&new_attr, sizeof(new_attr), attr);
	insn_cnt = new_attr.insn_cnt;
	insns = (void *)new_attr.insns;
	bpf_printk("insns: %p cnt %d\n", insns,
		   event->bpf_loaded_prog.insn_cnt);
	event->bpf_loaded_prog.insn_cnt = insn_cnt;
	event->bpf_loaded_prog.type = new_attr.prog_type;
	// copy name of the program
	*((__uint128_t *)&event->bpf_loaded_prog.prog_name) =
		*((__uint128_t *)&new_attr.prog_name);
	int insn_buffer_size =
		sizeof(event->bpf_loaded_prog.insns) >
				insn_cnt * sizeof(struct bpf_insn) ?
			insn_cnt * sizeof(struct bpf_insn) :
			sizeof(event->bpf_loaded_prog.insns);
	bpf_probe_read_user(event->bpf_loaded_prog.insns, insn_buffer_size,
			    insns);
	if (should_modify_program(&new_attr)) {
		struct bpf_insn trival_prog_insns[] = {
			BPF_MOV64_IMM(BPF_REG_0, 0),
			BPF_EXIT_INSN(),
		};

		// modify insns to make it trival
		new_attr.insn_cnt = 2;
		new_attr.func_info_rec_size = 0;
		new_attr.func_info = 0;
		new_attr.func_info_cnt = 0;

		new_attr.line_info_rec_size = 0;
		new_attr.line_info = 0;
		new_attr.line_info_cnt = 0;
		bpf_probe_write_user(insns, &trival_prog_insns,
				     sizeof(trival_prog_insns));
		bpf_probe_write_user(attr, &new_attr, sizeof(new_attr));
		// check whether write is success
		bpf_probe_read_user(&new_attr, sizeof(new_attr), attr);
		if (new_attr.insn_cnt != 2) {
			bpf_printk("write failed\n");
		}
	}
	/* emit event */
	bpf_ringbuf_submit(event, 0);
	return 0;
}

static int process_bpf_syscall_enter(struct trace_event_raw_sys_enter *ctx)
{
	enum bpf_cmd cmd = (enum bpf_cmd)ctx->args[0];
	union bpf_attr *attr = (union bpf_attr *)ctx->args[1];

	unsigned int size = (unsigned int)ctx->args[2];
	if (!attr || size < sizeof(*attr)) {
		return 0;
	}

	if (cmd == BPF_PROG_LOAD) {
		return process_bpf_prog_load_events(attr);
	}
	return 0;
}

SEC("tracepoint/syscalls/sys_enter_bpf")
int tracepoint__syscalls__sys_enter_bpf(struct trace_event_raw_sys_enter *ctx)
{
	if (!filter_target()) {
		return 0;
	}

	/* store arg info for later lookup */
	struct bpf_args_t args = {};
	args.cmd = (u32)ctx->args[0];
	args.attr = (void *)ctx->args[1];
	args.attr_size = (u32)ctx->args[2];

	u32 pid = bpf_get_current_pid_tgid() >> 32;
	bpf_map_update_elem(&bpf_param_start, &pid, &args, 0);

	process_bpf_syscall_enter(ctx);
	return 0;
}

SEC("tracepoint/syscalls/sys_exit_bpf")
int tracepoint__syscalls__sys_exit_bpf(struct trace_event_raw_sys_exit *ctx)
{
	struct event *event = NULL;
	struct bpf_args_t *ap = NULL;
	u32 pid = 0;

	if (!filter_target()) {
		return 0;
	}
	pid = bpf_get_current_pid_tgid() >> 32;
	ap = bpf_map_lookup_elem(&bpf_param_start, &pid);
	if (!ap)
		return 0; /* missed entry */

	event = fill_basic_event_info();
	if (!event) {
		return 0;
	}
	event->type = SYS_BPF;
	bpf_probe_read_user_str(&event->bpf_data.attr,
				sizeof(event->bpf_data.attr), ap->attr);
	event->bpf_data.attr_size = ap->attr_size;
	event->bpf_data.bpf_cmd = ap->cmd;
	event->bpf_data.ret = ctx->ret;

	/* emit event */
	bpf_ringbuf_submit(event, 0);
cleanup:
	bpf_map_delete_elem(&bpf_param_start, &pid);
	return 0;
}

static __always_inline int
process_perf_event_open_enter(struct trace_event_raw_sys_enter *ctx)
{
	struct perf_event_attr *attr = (struct perf_event_attr *)ctx->args[0];
	struct perf_event_attr empty_attr = {};
	if (!attr) {
		return 0;
	}
	// bpf_probe_write_user(attr, &empty_attr, sizeof(empty_attr));
	return 0;
}

struct perf_event_args_t {
	struct perf_event_attr *attr;
	int pid;
	int cpu;
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u32);
	__type(value, struct perf_event_args_t);
} perf_event_open_param_start SEC(".maps");

SEC("tracepoint/syscalls/sys_enter_perf_event_open")
int tracepoint__syscalls__sys_enter_perf_event_open(
	struct trace_event_raw_sys_enter *ctx)
{
	struct event *event;
	if (!filter_target()) {
		return 0;
	}
	/* store arg info for later lookup */

	/* store arg info for later lookup */
	struct perf_event_args_t args = {};
	args.attr = (void *)ctx->args[0];
	args.pid = (int)ctx->args[1];
	args.cpu = (int)ctx->args[2];

	u32 pid = bpf_get_current_pid_tgid() >> 32;
	bpf_map_update_elem(&perf_event_open_param_start, &pid, &args, 0);

	process_perf_event_open_enter(ctx);
	return 0;
}

SEC("tracepoint/syscalls/sys_exit_perf_event_open")
int tracepoint__syscalls__sys_exit_perf_event_open(
	struct trace_event_raw_sys_exit *ctx)
{
	struct event *event = NULL;
	struct perf_event_args_t *ap = NULL;
	u32 pid = 0;

	if (!filter_target()) {
		return 0;
	}
	pid = bpf_get_current_pid_tgid() >> 32;
	ap = bpf_map_lookup_elem(&perf_event_open_param_start, &pid);
	if (!ap)
		return 0; /* missed entry */

	/* event data */
	event = fill_basic_event_info();
	if (!event) {
		return 0;
	}
	event->type = SYS_PERF_EVENT_OPEN;

	bpf_probe_read_user_str(&event->perf_event_data.attr,
				sizeof(event->perf_event_data.attr), ap->attr);
	event->perf_event_data.pid = ap->pid;
	event->perf_event_data.cpu = ap->cpu;
	event->perf_event_data.ret = ctx->ret;

	/* emit event */
	bpf_ringbuf_submit(event, 0);
cleanup:
	bpf_map_delete_elem(&bpf_param_start, &pid);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
