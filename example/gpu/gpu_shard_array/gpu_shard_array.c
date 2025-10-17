// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#define _GNU_SOURCE
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <dlfcn.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include "./.output/gpu_shard_array.skel.h"

static volatile int exiting;

static void sig_handler(int sig)
{
	exiting = 1;
}

int main()
{
	struct gpu_shard_array_bpf *skel;
	int err;

	libbpf_set_strict_mode(LIBBPF_STRICT_ALL);
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	skel = gpu_shard_array_bpf__open();
	if (!skel) {
		fprintf(stderr, "open skel failed\n");
		return 1;
	}
	err = gpu_shard_array_bpf__load(skel);
	if (err) {
		fprintf(stderr, "load skel failed\n");
		goto cleanup;
	}
	err = gpu_shard_array_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "attach skel failed\n");
		goto cleanup;
	}

	// Observe-only: no server-side writeback; rely on agent/device-side
	// updates
	while (!exiting) {
		uint32_t key = 0;
		uint64_t value = 0;
		int mapfd = bpf_map__fd(skel->maps.counter);
		if (bpf_map_lookup_elem(mapfd, &key, &value) == 0) {
			printf("counter[0]=%lu\n", (unsigned long)value);
		}
		// value = 0;
		sleep(1);
	}

cleanup:
	gpu_shard_array_bpf__destroy(skel);
	return err != 0;
}
