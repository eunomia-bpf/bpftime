#include <stdio.h>
#include <unistd.h>
#include <bpf/libbpf.h>
#include "perthreadRuntimeDist.skel.h"
#include "perthreadRuntimeDist.h"

static void handle_event(void *ctx, int cpu, void *data, __u32 size)
{
    struct event_t *e = data;
    printf("[CPU %d] tid=%u cycles=%llu ns\n",
           cpu, e->tid, e->cycles);
}

static void handle_lost(void *ctx, int cpu, __u64 lost)
{
    printf("LOST %llu events on CPU %d\n", lost, cpu);
}

int main()
{
    struct perthreadRuntimeDist_bpf *skel;
    struct perf_buffer *pb;
    int events_fd;

    skel = perthreadRuntimeDist_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    if (perthreadRuntimeDist_bpf__load(skel)) {
        fprintf(stderr, "Failed to load BPF skeleton\n");
        return 1;
    }

    if (perthreadRuntimeDist_bpf__attach(skel)) {
        fprintf(stderr, "Failed to attach BPF skeleton\n");
        return 1;
    }

    printf("BPF attached successfully\n");

    events_fd = bpf_map__fd(skel->maps.events);
    pb = perf_buffer__new(events_fd, 16 /*buffer pages*/,
                         handle_event, handle_lost, NULL, NULL);

    if (!pb) {
        fprintf(stderr, "Failed to open perf buffer\n");
        return 1;
    }

    printf("Collecting data...\n");

    while (1) {
        int err = perf_buffer__poll(pb, 100 /*ms*/);
        if (err < 0)
            printf("perf_buffer__poll() error %d\n", err);
    }

    return 0;
}
