// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#include "./.output/multi_gpu_probe.skel.h"
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static volatile bool exiting;

static void sig_handler(int sig)
{
	(void)sig;
	exiting = true;
}

static int print_counts(struct multi_gpu_probe_bpf *skel)
{
	int fd = bpf_map__fd(skel->maps.launch_count);
	uint32_t key;
	uint32_t previous;
	uint32_t *previous_ptr = NULL;

	while (bpf_map_get_next_key(fd, previous_ptr, &key) == 0) {
		uint64_t count;
		if (bpf_map_lookup_elem(fd, &key, &count) != 0)
			return -errno;
		printf("GPU %u: %llu vectorAdd launch(es)\n", key,
		       (unsigned long long)count);
		previous = key;
		previous_ptr = &previous;
	}
	return errno == ENOENT ? 0 : -errno;
}

int main(void)
{
	struct multi_gpu_probe_bpf *skel;
	int err;

	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);
	skel = multi_gpu_probe_bpf__open_and_load();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}
	err = multi_gpu_probe_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton: %s\n",
			strerror(-err));
		goto cleanup;
	}

	while (!exiting) {
		err = print_counts(skel);
		if (err)
			break;
		sleep(2);
	}

cleanup:
	multi_gpu_probe_bpf__destroy(skel);
	return err < 0 ? -err : err;
}
