// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
/* Copyright (c) 2020 Facebook */
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "bootstrap.h"

char LICENSE[] SEC("license") = "Dual BSD/GPL";

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 8192);
	__type(key, pid_t);
	__type(value, struct event);
} exec_start SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 256 * 1024);
} rb SEC(".maps");

static inline int handle_exec()
{
	struct event *e;
	pid_t pid;

	/* remember time exec() was executed for this PID */
	pid = bpf_get_current_pid_tgid() >> 32;
	e = bpf_map_lookup_elem(&exec_start, &pid);
	if (!e) {
		struct event new_e = { 1, 0 };
		bpf_map_update_elem(&exec_start, &pid, &new_e, BPF_ANY);
		return 0;
	}
	e->enter_count++;
	bpf_map_update_elem(&exec_start, &pid, &e, BPF_ANY);
	return 0;
}

SEC("uprobe//home/yunwei/bpftime/documents/frida-gum-uprobe/bin/victim:my_test_func")
int handle_open(struct pt_regs *ctx)
{
	return handle_exec();
}

// SEC("uprobe//home/yunwei/bpftime/documents/frida-gum-uprobe/bin/victim:my_close")
// int handle_close(struct pt_regs *ctx)
// {
// 	return handle_exec();
// }

static inline int handle_exit()
{
	struct event *e;
	pid_t pid;

	pid = bpf_get_current_pid_tgid() >> 32;
	e = bpf_map_lookup_elem(&exec_start, &pid);
	if (!e) {
		return 0;
	}
	e->exit_count++;
	bpf_map_update_elem(&exec_start, &pid, &e, BPF_ANY);

	return 0;
}

SEC("uretprobe//home/yunwei/bpftime/documents/frida-gum-uprobe/bin/victim:my_test_func")
int handle_open_exit() {
	return handle_exit();
}

// SEC("uretprobe//home/yunwei/bpftime/documents/frida-gum-uprobe/bin/victim:my_close")
// int handle_close_exit() {
// 	return handle_exit();
// }
