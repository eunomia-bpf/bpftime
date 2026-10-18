// SPDX-License-Identifier: GPL-2.0
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "./get_stack_id_example.h"

#define MAX_STACK_DEPTH 20

struct {
	__uint(type, BPF_MAP_TYPE_STACK_TRACE);
	__uint(key_size, sizeof(u32));
	__uint(value_size, MAX_STACK_DEPTH * sizeof(u64));
	__uint(max_entries, 1000);
} stack_traces SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 256 * 1024);
} rb SEC(".maps");

SEC("uprobe/libc.so.6:malloc")
int malloc_enter(struct pt_regs *ctx)
{
	struct event *e;

	e = bpf_ringbuf_reserve(&rb, sizeof(*e), 0);
	if (!e) {
		return 0;
	}

	e->pid = bpf_get_current_pid_tgid() >> 32;
	e->stack_id = bpf_get_stackid(ctx, &stack_traces, BPF_F_USER_STACK);
	e->operation = MALLOC_ENTER;
	bpf_ringbuf_submit(e, 0);

	return 0;
}

SEC("uprobe/libc.so.6:free")
int free_enter(struct pt_regs *ctx)
{
	struct event *e;

	e = bpf_ringbuf_reserve(&rb, sizeof(*e), 0);
	if (!e) {
		return 0;
	}

	e->pid = bpf_get_current_pid_tgid() >> 32;
	e->stack_id = bpf_get_stackid(ctx, &stack_traces, BPF_F_USER_STACK);
	e->operation = FREE_ENTER;
	bpf_ringbuf_submit(e, 0);

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
