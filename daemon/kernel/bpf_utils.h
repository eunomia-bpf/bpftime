#ifndef BPF_UTILS_H
#define BPF_UTILS_H

#include <vmlinux.h>
#include "bpf_defs.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "../bpf_tracer_event.h"

// filter bpf program pid
const volatile int target_pid = 0;
// current bpf program pid, avoid breaking current process
const volatile int current_pid = 0;
// disable modify bpf program
const volatile bool disable_modify = 0;

const volatile int uprobe_perf_type = 0;
const volatile int kprobe_perf_type = 0;

// print event to userspace
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 256 * 1024);
} rb SEC(".maps");

// pid & fd for all bpf related fds
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u64);
	__type(value, u64);
} bpf_fd_map SEC(".maps");

#define PID_MASK_FOR_PFD 0xffffffff00000000
#define FD_MASK_FOR_PFD 0x00000000ffffffff
#define MAKE_PFD(pid, fd) (((u64)pid << 32) | fd)

static __always_inline bool is_bpf_fd(u32 fd) {
	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u64 key = MAKE_PFD(pid, fd);
	u64 *pfd = bpf_map_lookup_elem(&bpf_fd_map, &key);
	if (!pfd) {
		return false;
	}
	return true; 
}

static __always_inline void set_bpf_fd_if_positive(u32 fd) {
	if (fd < 0) {
		return;
	}
	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u64 key = MAKE_PFD(pid, fd);
	bpf_map_update_elem(&bpf_fd_map, &key, &key, 0);
}

static __always_inline void clear_bpf_fd(int fd) {
	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u64 key = MAKE_PFD(pid, fd);
	bpf_map_delete_elem(&bpf_fd_map, &key);
}

static __always_inline bool filter_target(void)
{
    u64 pid = bpf_get_current_pid_tgid() >> 32;
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

static __always_inline struct event* fill_basic_event_info(void) {
    struct event *event = bpf_ringbuf_reserve(&rb, sizeof(struct event), 0);
    if (!event) {
        return NULL;
    }
    event->pid = bpf_get_current_pid_tgid() >> 32;
	bpf_get_current_comm(&event->comm, sizeof(event->comm));
    return event;
}

#endif // BPF_UTILS_H
