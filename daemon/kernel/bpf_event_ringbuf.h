/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef BPF_UTILS_H
#define BPF_UTILS_H

#include <vmlinux.h>
#include "bpf_defs.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "../bpf_tracer_event.h"
#include "bpf_kernel_config.h"
#include "bpf_pid_fd_map.h"

// print event to userspace
struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 256 * 1024);
} rb SEC(".maps");

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
