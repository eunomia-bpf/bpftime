#ifndef BPFTIME_HANDLE_EVENT_HPP
#define BPFTIME_HANDLE_EVENT_HPP

#include "../bpf_tracer_event.h"
#include "daemon_config.hpp"

namespace bpftime {

class bpf_event_handler {
    struct env config;
    
    int handle_close_event(const struct event *e);
    int handle_bpf_event(const struct event *e);
    int handle_open_events(const struct event *e);
    int handle_perf_event(const struct event *e);
    int handle_load_bpf_prog_event(const struct event *e);
public:
    int handle_event(const struct event *e);

    bpf_event_handler(struct env config);
};

int determine_kprobe_perf_type(void);

int determine_uprobe_perf_type(void);

} // namespace bpftime

#endif // BPFTIME_HANDLE_EVENT_HPP
