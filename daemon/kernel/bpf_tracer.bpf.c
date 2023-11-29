/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "../bpf_tracer_event.h"
#include "bpf_event_ringbuf.h"
#include "bpf_obj_id_fd_map.h"

struct open_args_t {
	const char *fname;
	int flags;
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u64);
	__type(value, u32);
} whitelist_hook_addr SEC(".maps");

const volatile int enable_whitelist = 0;

__always_inline int can_hook_uprobe_at(u64 addr)
{
	int ok = 0;
	if (!enable_whitelist)
		ok = 1;
	else {
		ok = bpf_map_lookup_elem(&whitelist_hook_addr, &addr) != NULL;
	}
	bpf_printk("Testing: addr=%llx, ok=%d", (unsigned long long)addr, ok);
	return ok;
}

// track open syscall args
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u64);
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

	u64 pid_tgid = bpf_get_current_pid_tgid();
	bpf_map_update_elem(&open_param_start, &pid_tgid, &args, 0);
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

	u64 pid_tgid = bpf_get_current_pid_tgid();
	bpf_map_update_elem(&open_param_start, &pid_tgid, &args, 0);
	return 0;
}

static __always_inline int trace_exit_open(struct trace_event_raw_sys_exit *ctx)
{
	struct event *event = NULL;
	struct open_args_t *ap = NULL;
	u64 pid_tgid = 0;

	if (!filter_target()) {
		return 0;
	}
	pid_tgid = bpf_get_current_pid_tgid();
	ap = bpf_map_lookup_elem(&open_param_start, &pid_tgid);
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
	bpf_map_delete_elem(&open_param_start, &pid_tgid);
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
	union bpf_attr attr;
	u32 attr_size;
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u64);
	__type(value, struct bpf_args_t);
} bpf_param_start SEC(".maps");

static bool should_modify_program(union bpf_attr *new_attr)
{
	if (new_attr->insn_cnt < 2 || !enable_replace_prog) {
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
	union bpf_attr new_attr = { 0 };
	bpf_probe_read(&new_attr, sizeof(new_attr), attr);

	insn_cnt = BPF_CORE_READ_USER(attr, insn_cnt);
	insns = (void *)BPF_CORE_READ_USER(attr, insns);
	bpf_printk("insns: %p cnt %d\n", insns, insn_cnt);

	insns_data.code_len = insn_cnt;
	unsigned int read_len =
		sizeof(insns_data.code) > insn_cnt * sizeof(struct bpf_insn) ?
			insn_cnt * sizeof(struct bpf_insn) :
			sizeof(insns_data.code);
	bpf_probe_read_user(&insns_data.code, read_len, insns);
	u64 pid_tgid = bpf_get_current_pid_tgid();
	bpf_map_update_elem(&bpf_prog_insns_map, &pid_tgid, &insns_data,
			    BPF_NOEXIST);
	if (submit_bpf_events) {
		/* event data */
		struct event *event = NULL;
		event = fill_basic_event_info();
		if (!event) {
			return 0;
		}
		event->type = BPF_PROG_LOAD_EVENT;
		event->bpf_loaded_prog.insn_cnt = insn_cnt;
		event->bpf_loaded_prog.type =
			BPF_CORE_READ_USER(attr, prog_type);
		event->bpf_loaded_prog.insns_ptr = (u64)insns;
		// copy name of the program
		*((__uint128_t *)&event->bpf_loaded_prog.prog_name) =
			*((__uint128_t *)new_attr.prog_name);
		/* emit event */
		bpf_ringbuf_submit(event, 0);
	}

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
			bpf_printk("bpftime: write failed\n");
		}
	}
	return 0;
}

static int process_bpf_syscall_enter(struct trace_event_raw_sys_enter *ctx)
{
	enum bpf_cmd cmd = (enum bpf_cmd)ctx->args[0];
	union bpf_attr *attr = (union bpf_attr *)ctx->args[1];

	unsigned int size = (unsigned int)ctx->args[2];
	if (!attr) {
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
		bpf_printk("bpftime: Skipping sys enter bpf");
		return 0;
	}

	/* store arg info for later lookup */
	struct bpf_args_t args = {};
	args.cmd = (u32)ctx->args[0];
	bpf_probe_read_user(&args.attr, sizeof(args.attr),
			    (void *)ctx->args[1]);
	args.attr_size = (u32)ctx->args[2];

	u64 pid_tgid = bpf_get_current_pid_tgid();
	bpf_map_update_elem(&bpf_param_start, &pid_tgid, &args, 0);

	process_bpf_syscall_enter(ctx);
	return 0;
}

static __always_inline struct event *
get_ringbuf_sys_exit_bpf_event(struct bpf_args_t *ap, int ret)
{
	struct event *event = fill_basic_event_info();
	if (!event) {
		return 0;
	}
	event->type = SYS_BPF;
	bpf_probe_read(&event->bpf_data.attr, sizeof(event->bpf_data.attr),
		       &ap->attr);
	event->bpf_data.attr_size = ap->attr_size;
	event->bpf_data.bpf_cmd = ap->cmd;
	event->bpf_data.ret = ret;
	return event;
}

static int process_bpf_syscall_exit(struct bpf_args_t *ap, int ret)
{
	if (!ap) {
		return 0;
	}
	unsigned int cmd = ap->cmd;
	if (ret >= 0) {
		switch (cmd) {
		case BPF_PROG_LOAD: {
			u64 pid_tgid = bpf_get_current_pid_tgid();
			u32 *id_ptr = (u32 *)bpf_map_lookup_elem(
				&bpf_progs_new_fd_args_map, &pid_tgid);
			if (!id_ptr)
				return 0; /* missed entry */
			// bpf_map_delete_elem(&bpf_progs_new_fd_args_map,
			// 		    &pid_tgid);
			struct bpf_fd_data data = { .type = BPF_FD_TYPE_PROG,
						    .kernel_id = *id_ptr };

			set_bpf_fd_data(ret, &data);
			bpf_printk(
				"bpftime: bpf_prog_load exit id: %d fd: %d\n",
				*id_ptr, ret);
			break;
		}

		case BPF_MAP_CREATE: {
			u64 pid_tgid = bpf_get_current_pid_tgid();
			u32 *id_ptr = (u32 *)bpf_map_lookup_elem(
				&bpf_map_new_fd_args_map, &pid_tgid);
			if (!id_ptr)
				return 0; /* missed entry */
			u32 id = *id_ptr;
			// bpf_map_delete_elem(&bpf_map_new_fd_args_map,
			// 		    &pid_tgid);
			struct bpf_fd_data data = { .type = BPF_FD_TYPE_MAP,
						    .kernel_id = id };
			// record map fd and id
			set_bpf_fd_data(ret, &data);
			bpf_printk(
				"bpftime: bpf_map_create exit id: %d fd: %d\n",
				id, ret);
			if (submit_bpf_events) {
				// submit event to user for record
				struct event *event =
					get_ringbuf_sys_exit_bpf_event(ap, ret);
				// Make verifier happy
				if (!event)
					return 0;
				event->bpf_data.map_id = (int)id;
				bpf_ringbuf_submit(event, 0);
			}
			break;
		}
		case BPF_LINK_CREATE: {
			u64 pid_tgid = bpf_get_current_pid_tgid();
			u32 *id_ptr = (u32 *)bpf_map_lookup_elem(
				&bpf_progs_new_fd_args_map, &pid_tgid);
			if (!id_ptr) {
				bpf_printk(
					"bpftime: Unable to get prog id when creating link");
				return 0; /* missed entry */
			}
			u32 id = *id_ptr;
			// bpf_map_delete_elem(&bpf_map_new_fd_args_map,
			// 		    &pid_tgid);
			struct bpf_fd_data data = { .type = BPF_FD_TYPE_OTHERS,
						    .kernel_id = id };
			set_bpf_fd_data(ret, &data);
			bpf_printk("bpftime: bpf_link_create");
			bpf_printk("bpftime: Submitting link creation");
			struct event *event =
				get_ringbuf_sys_exit_bpf_event(ap, ret);
			if (!event)
				return 0;
			bpf_ringbuf_submit(event, 0);
			break;
		}
		case BPF_BTF_LOAD: {
			struct bpf_fd_data data = { .type = BPF_FD_TYPE_OTHERS,
						    .kernel_id = 0 };
			set_bpf_fd_data(ret, &data);
			if (submit_bpf_events) {
				struct event *event =
					get_ringbuf_sys_exit_bpf_event(ap, ret);
				if (!event)
					return 0;
				bpf_ringbuf_submit(event, 0);
			}
			break;
		}
		default:
			break;
		}
	}
	return 0;
}

SEC("tracepoint/syscalls/sys_exit_bpf")
int tracepoint__syscalls__sys_exit_bpf(struct trace_event_raw_sys_exit *ctx)
{
	struct bpf_args_t *ap = NULL;

	if (!filter_target()) {
		return 0;
	}
	u64 pid_tgid = bpf_get_current_pid_tgid();
	ap = bpf_map_lookup_elem(&bpf_param_start, &pid_tgid);
	if (!ap)
		return 0; /* missed entry */

	process_bpf_syscall_exit(ap, ctx->ret);

cleanup:
	bpf_map_delete_elem(&bpf_param_start, &pid_tgid);
	return 0;
}

// uprobe path buffer for bpf_probe_read_user_str
char old_uprobe_path[PATH_LENTH] = "\0";
// avoid type mismatch in userspace program
// struct perf_event_attr new_attr = {}; will result in compile error
char new_attr_buffer[sizeof(struct perf_event_attr)] = "\0";

static __always_inline int
process_perf_event_open_enter(struct trace_event_raw_sys_enter *ctx)
{
	struct perf_event_attr *attr = (struct perf_event_attr *)ctx->args[0];
	if (!attr) {
		return 0;
	}
	bpf_probe_read_user(&new_attr_buffer, sizeof(new_attr_buffer), attr);
	struct perf_event_attr *new_attr_pointer =
		(struct perf_event_attr *)&new_attr_buffer;
	if (new_attr_pointer->type == uprobe_perf_type) {
		// found uprobe
		if (enable_replace_uprobe) {
			if (can_hook_uprobe_at(
				    new_attr_pointer->probe_offset)) {
				u64 old_offset = new_attr_pointer->probe_offset;
				new_attr_pointer->probe_offset = 0;
				long size = bpf_probe_read_user_str(
					old_uprobe_path,
					sizeof(old_uprobe_path),
					(void *)new_attr_pointer->uprobe_path);
				if (size <= 0) {
					// no uprobe path
					return 0;
				}
				if (size > PATH_LENTH) {
					size = PATH_LENTH;
				}
				bpf_printk("bpftime: new uprobe path: %s",
					   new_uprobe_path);
				bpf_probe_write_user(
					(void *)new_attr_pointer->uprobe_path,
					&new_uprobe_path, (size_t)size);
				bpf_probe_write_user(attr, new_attr_pointer,
						     sizeof(*new_attr_pointer));
				// This probe creation request should be
				// executed in userspace
				bpf_printk(
					"Send perf event at offset %lx to userspace",
					old_offset);
				return 1;
			} else {
				return 0;
			}
		} else {
			return 1;
		}
		return 0;
	}
	return 0;
}

struct perf_event_args_t {
	struct perf_event_attr attr;
	int pid;
	int cpu;

	// we may modify the offset and name, so we keep it here
	char name_or_path[NAME_MAX];
	// original offset, if we modified
	u64 orig_offset;
	// Whether to send thie creation to bpftime_daemon?
	int send_to_daemon;
	void *path_buf_user;
	struct perf_event_attr *attr_user;
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u64);
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
	struct perf_event_args_t args = {};
	bpf_probe_read_user(&args.attr, sizeof(args.attr),
			    (void *)ctx->args[0]);
	args.pid = (int)ctx->args[1];
	args.cpu = (int)ctx->args[2];
	args.attr_user = (struct perf_event_attr *)ctx->args[0];
	args.path_buf_user = (void *)args.attr.uprobe_path;
	bpf_probe_read_user_str(args.name_or_path, PATH_LENTH,
				(const void *)args.attr.uprobe_path);
	args.orig_offset = args.attr.probe_offset;
	bpf_printk("Received path: %s", args.name_or_path);
	u64 pid_tgid = bpf_get_current_pid_tgid();
	if (process_perf_event_open_enter(ctx) == 1) {
		args.send_to_daemon = 1;
	} else {
		args.send_to_daemon = 0;
	}
	bpf_map_update_elem(&perf_event_open_param_start, &pid_tgid, &args, 0);

	return 0;
}

SEC("tracepoint/syscalls/sys_exit_perf_event_open")
int tracepoint__syscalls__sys_exit_perf_event_open(
	struct trace_event_raw_sys_exit *ctx)
{
	struct event *event = NULL;
	struct perf_event_args_t *ap = NULL;
	struct bpf_fd_data data = { .type = BPF_FD_TYPE_PERF, .kernel_id = 0 };

	if (!filter_target()) {
		return 0;
	}
	u64 pid_tgid = bpf_get_current_pid_tgid();
	ap = bpf_map_lookup_elem(&perf_event_open_param_start, &pid_tgid);
	if (!ap)
		return 0; /* missed entry */

	if (ap->send_to_daemon) {
		set_bpf_fd_data(ctx->ret, &data);

		/* event data */
		event = fill_basic_event_info();
		if (!event) {
			return 0;
		}
		event->type = SYS_PERF_EVENT_OPEN;

		bpf_probe_read(&event->perf_event_data.attr,
			       sizeof(event->perf_event_data.attr), &ap->attr);
		event->perf_event_data.pid = ap->pid;
		event->perf_event_data.cpu = ap->cpu;
		event->perf_event_data.ret = ctx->ret;
		bpf_probe_read(event->perf_event_data.name_or_path, PATH_LENTH,
			       ap->name_or_path);

		/* emit event */
		bpf_ringbuf_submit(event, 0);
		bpf_printk("Accept perf event creation at program %s to daemon",
			   ap->name_or_path);
		// Revert changes
		bpf_probe_write_user(ap->path_buf_user, ap->name_or_path,
				     sizeof(ap->name_or_path));
		bpf_probe_write_user(&ap->attr_user->probe_offset,
				     &ap->orig_offset, sizeof(ap->orig_offset));
	} else {
		bpf_printk("Reject perf event creation at program %s to daemon",
			   ap->name_or_path);
	}
cleanup:
	bpf_map_delete_elem(&perf_event_open_param_start, &pid_tgid);
	return 0;
}

SEC("tracepoint/syscalls/sys_enter_close")
int tracepoint__syscalls__sys_enter_close(struct trace_event_raw_sys_enter *ctx)
{
	struct event *event = NULL;

	if (!filter_target()) {
		return 0;
	}
	int fd = (int)ctx->args[0];
	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u64 key = MAKE_PFD(pid, fd);
	struct bpf_fd_data *pfd = bpf_map_lookup_elem(&bpf_fd_map, &key);
	if (!pfd) {
		return 0;
	}
	if (!submit_bpf_events && pfd->type != BPF_FD_TYPE_PERF) {
		// ignore not perf close event
		return 0;
	}
	/* event data */
	event = fill_basic_event_info();
	if (!event) {
		return 0;
	}
	event->type = SYS_CLOSE;

	event->close_data.fd = fd;
	bpf_probe_read((void *)&(event->close_data.fd_data),
		       sizeof(struct bpf_fd_data), (void *)pfd);
	/* emit event */
	bpf_ringbuf_submit(event, 0);
	clear_bpf_fd(fd);
	return 0;
}

struct ioctl_args_t {
	int fd;
	unsigned long req;
	int data;
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u64);
	__type(value, struct ioctl_args_t);
} ioctl_param_start SEC(".maps");

SEC("tracepoint/syscalls/sys_enter_ioctl")
int tracepoint__syscalls__sys_enter_ioctl(struct trace_event_raw_sys_enter *ctx)
{
	struct event *event;
	if (!filter_target()) {
		return 0;
	}
	int fd = (int)ctx->args[0];
	if (!is_bpf_fd(fd)) {
		return 0;
	}
	/* store arg info for later lookup */
	struct ioctl_args_t args = {};
	args.fd = fd;
	args.req = (int)ctx->args[1];
	args.data = (int)ctx->args[2];

	u64 pid_tgid = bpf_get_current_pid_tgid();
	bpf_map_update_elem(&ioctl_param_start, &pid_tgid, &args, 0);
	return 0;
}

SEC("tracepoint/syscalls/sys_exit_ioctl")
int tracepoint__syscalls__sys_exit_ioctl(struct trace_event_raw_sys_exit *ctx)
{
	struct event *event = NULL;
	struct ioctl_args_t *ap = NULL;

	if (!filter_target()) {
		return 0;
	}
	u64 pid_tgid = bpf_get_current_pid_tgid();
	ap = bpf_map_lookup_elem(&ioctl_param_start, &pid_tgid);
	if (!ap)
		return 0; /* missed entry */

	/* event data */
	event = fill_basic_event_info();
	if (!event) {
		return 0;
	}
	event->type = SYS_IOCTL;

	event->ioctl_data.data = ap->data;
	event->ioctl_data.fd = ap->fd;
	event->ioctl_data.req = ap->req;
	event->ioctl_data.ret = ctx->ret;

	if (PERF_EVENT_IOC_SET_BPF == ap->req) {
		u32 pid = bpf_get_current_pid_tgid() >> 32;
		u64 key = MAKE_PFD(pid, ap->data);
		struct bpf_fd_data *data =
			bpf_map_lookup_elem(&bpf_fd_map, &key);
		if (data) {
			event->ioctl_data.bpf_prog_id = data->kernel_id;
		} else {
			event->ioctl_data.bpf_prog_id = 0;
		}
	} else {
		event->ioctl_data.bpf_prog_id = 0;
	}

	/* emit event */
	bpf_ringbuf_submit(event, 0);
cleanup:
	bpf_map_delete_elem(&ioctl_param_start, &pid_tgid);
	return 0;
}

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 8192);
	__type(key, pid_t);
	__type(value, struct event);
} exec_start SEC(".maps");

SEC("tp/sched/sched_process_exec")
int handle_exec(struct trace_event_raw_sched_process_exec *ctx)
{
	struct task_struct *task;
	unsigned fname_off;
	struct event e = { 0 };
	pid_t pid;
	u64 ts;

	/* remember time exec() was executed for this PID */
	pid = bpf_get_current_pid_tgid() >> 32;
	e.exec_data.time_ns = bpf_ktime_get_ns();

	/* fill out the sample with data */
	task = (struct task_struct *)bpf_get_current_task();

	e.type = EXEC_EXIT;
	e.exec_data.exit_event = false;
	e.pid = pid;
	e.exec_data.ppid = BPF_CORE_READ(task, real_parent, tgid);
	bpf_get_current_comm(&e.comm, sizeof(e.comm));

	fname_off = ctx->__data_loc_filename & 0xFFFF;
	bpf_probe_read_str(&e.exec_data.filename, sizeof(e.exec_data.filename),
			   (void *)ctx + fname_off);

	/* successfully submit it to user-space for post-processing */
	bpf_map_update_elem(&exec_start, &pid, &e, BPF_ANY);
	return 0;
}

SEC("tp/sched/sched_process_exit")
int handle_exit(struct trace_event_raw_sched_process_template *ctx)
{
	struct task_struct *task;
	struct event *e;
	pid_t pid, tid;
	u64 id, ts, *start_ts, duration_ns = 0;

	/* get PID and TID of exiting thread/process */
	id = bpf_get_current_pid_tgid();
	pid = id >> 32;
	tid = (u32)id;

	/* ignore thread exits */
	if (pid != tid)
		return 0;

	/* if we recorded start of the process, calculate lifetime duration */
	e = bpf_map_lookup_elem(&exec_start, &pid);
	if (!e)
		return 0;
	bpf_map_delete_elem(&exec_start, &pid);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
