#ifndef __PERTHREAD_RUNTIME_DIST_H__
#define __PERTHREAD_RUNTIME_DIST_H__

struct event_t {
    __u32 tid;
    __u64 cycles;
};

#endif
