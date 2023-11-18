/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef BPFTIME_BPF_PID_FD_MAP
#define BPFTIME_BPF_PID_FD_MAP

#include <vmlinux.h>
#include "bpf_defs.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "../bpf_tracer_event.h"

// pid & fd for all bpf related fds
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 10240);
	__type(key, u64);
	__type(value, struct bpf_fd_data);
} bpf_fd_map SEC(".maps");

// is bpf fd in bpf_fd_map
static __always_inline bool is_bpf_fd(u32 fd) {
	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u64 key = MAKE_PFD(pid, fd);
	void *pfd = bpf_map_lookup_elem(&bpf_fd_map, &key);
	return pfd != NULL; 
}

static __always_inline void set_bpf_fd_data(u32 fd, struct bpf_fd_data *data) {
	if (fd < 0) {
		return;
	}
	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u64 key = MAKE_PFD(pid, fd);
	bpf_map_update_elem(&bpf_fd_map, &key, data, 0);
}

static __always_inline void clear_bpf_fd(int fd) {
	u32 pid = bpf_get_current_pid_tgid() >> 32;
	u64 key = MAKE_PFD(pid, fd);
	bpf_map_delete_elem(&bpf_fd_map, &key);
}


#endif // BPFTIME_BPF_PID_FD_MAP
