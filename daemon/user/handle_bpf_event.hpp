/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef BPFTIME_HANDLE_EVENT_HPP
#define BPFTIME_HANDLE_EVENT_HPP

#include "../bpf_tracer_event.h"
#include "daemon_config.hpp"
#include "bpftime_driver.hpp"
#include <map>
#include <cstdint>

namespace bpftime {

class bpf_event_handler {
    int current_pid = 0;
    int uprobe_type = -1;
    int kprobe_type = -1;

    struct daemon_config config;
    bpftime_driver &driver;
    
    int handle_close_event(const struct event *e);
    int handle_bpf_event(const struct event *e);
    int handle_open_events(const struct event *e);
    int handle_perf_event_open(const struct event *e);
    int handle_load_bpf_prog_event(const struct event *e);
    int handle_ioctl(const struct event *e);

    int handle_exec_and_exit(const struct event *e);
public:
    // callback function for bpf events in ring buffer
    int handle_event(const struct event *e);

    bpf_event_handler(struct daemon_config config, bpftime_driver &driver);
};

// determine the perf type for kprobe, exit if failed
int determine_kprobe_perf_type(void);

// determine the perf type for uprobe, exit if failed
int determine_uprobe_perf_type(void);

int determine_uprobe_retprobe_bit();

} // namespace bpftime

#endif // BPFTIME_HANDLE_EVENT_HPP
