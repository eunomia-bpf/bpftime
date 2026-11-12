#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "cuda_api_trace.skel.h"

static volatile bool exiting;

static void sig_handler(int sig)
{
    exiting = true;
}

static int print_counts(struct cuda_api_trace_bpf *skel)
{
    uint32_t key;
    uint32_t *previous = NULL;
    uint64_t value;
    int fd = bpf_map__fd(skel->maps.cuda_launch_calls_total);

    while (bpf_map_get_next_key(fd, previous, &key) == 0) {
        if (bpf_map_lookup_elem(fd, &key, &value) != 0) {
            return -errno;
        }
        printf("pid=%u cudaLaunchKernel calls=%llu\n", key,
               (unsigned long long)value);
        previous = &key;
    }
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv)
{
    const char *library = argc > 1 ? argv[1] : "/usr/local/cuda/lib64/libcudart.so";
    struct cuda_api_trace_bpf *skel = cuda_api_trace_bpf__open();
    struct bpf_link *link = NULL;
    int err = 0;

    if (!skel) {
        fprintf(stderr, "failed to open CUDA API trace skeleton\n");
        return 1;
    }
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    if ((err = cuda_api_trace_bpf__load(skel)) != 0) {
        fprintf(stderr, "failed to load CUDA API trace skeleton: %d\n", err);
        goto cleanup;
    }

    LIBBPF_OPTS(bpf_uprobe_opts, opts,
        .func_name = "cudaLaunchKernel",
        .retprobe = false,
    );
    link = bpf_program__attach_uprobe_opts(
        skel->progs.count_cuda_launch, -1, library, 0, &opts);
    if (!link) {
        err = -errno;
        fprintf(stderr, "failed to attach to %s:cudaLaunchKernel: %s\n",
                library, strerror(errno));
        goto cleanup;
    }
    printf("attached to %s:cudaLaunchKernel\n", library);
    while (!exiting) {
        sleep(1);
        if ((err = print_counts(skel)) != 0) {
            goto cleanup;
        }
    }

cleanup:
    bpf_link__destroy(link);
    cuda_api_trace_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
