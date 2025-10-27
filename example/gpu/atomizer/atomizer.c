// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/* Copyright (c) 2020 Facebook */
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <unistd.h>
#include <stdlib.h>
#include "./.output/atomizer.skel.h"
#include <inttypes.h>
#define warn(...) fprintf(stderr, __VA_ARGS__)

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

static void sig_handler(int sig)
{
	exiting = true;
}

static int setup_partition_maps(struct atomizer_bpf *obj)
{
	int err = 0;
	{
		uint32_t key = 0;
		uint64_t value = 2;
		int fd = bpf_map__fd(obj->maps.partition_num_map);
		err = bpf_map_update_elem(fd, &key, &value, BPF_ANY);
		if (err) {
			warn("bpf_map_update_elem failed: %s\n",
			     strerror(errno));
			return err;
		}
		printf("	partition_num is set to: %" PRIu64 "\n", value);
	}
	{
		uint32_t key = 0;
		uint64_t value = 1;
		int fd = bpf_map__fd(obj->maps.partition_index_map);
		err = bpf_map_update_elem(fd, &key, &value, BPF_ANY);
		if (err) {
			warn("bpf_map_update_elem failed: %s\n",
			     strerror(errno));
			return err;
		}
		printf("	partition_idx is set to: %" PRIu64 "\n", value);
	}
	return err;
}

int main(int argc, char **argv)
{
	struct atomizer_bpf *skel;
	int err;

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = atomizer_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}

	/* Load & verify BPF programs */
	err = atomizer_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}

	setup_partition_maps(skel);

	err = atomizer_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton\n");
		goto cleanup;
	}

	while (!exiting) {
		sleep(1);
	}
cleanup:
	/* Clean up */
	atomizer_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
