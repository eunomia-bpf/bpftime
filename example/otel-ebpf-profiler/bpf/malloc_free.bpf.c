// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

enum allocation_event {
	EVENT_MALLOC = 1,
	EVENT_FREE = 2,
};

struct event_key {
	u32 pid;
	u32 event;
};

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, struct event_key);
	__type(value, u64);
} allocation_events SEC(".maps");

static __always_inline void increment_event(u32 pid, u32 event)
{
	struct event_key key = {
		.pid = pid,
		.event = event,
	};
	u64 zero = 0;
	u64 *count = bpf_map_lookup_elem(&allocation_events, &key);

	if (!count) {
		bpf_map_update_elem(&allocation_events, &key, &zero,
				    BPF_NOEXIST);
		count = bpf_map_lookup_elem(&allocation_events, &key);
		if (!count) {
			return;
		}
	}

	zero = *count + 1;
	bpf_map_update_elem(&allocation_events, &key, &zero, BPF_EXIST);
}

SEC("kprobe/generic")
int kprobe__generic(struct pt_regs *ctx)
{
	u64 cookie = bpf_get_attach_cookie(ctx);
	u32 event = cookie == EVENT_FREE ? EVENT_FREE : EVENT_MALLOC;
	u32 pid = bpf_get_current_pid_tgid() >> 32;

	increment_event(pid, event);
	return 0;
}

char LICENSE[] SEC("license") = "Dual BSD/GPL";
