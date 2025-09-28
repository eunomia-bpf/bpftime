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
#include "./.output/cuda_probe.skel.h"
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

static int print_stat(struct cuda_probe_bpf *obj, uint64_t thread_count)
{
	time_t t;
	struct tm *tm;
	char ts[16];
	uint32_t key;
	int err = 0;
	int fd = bpf_map__fd(obj->maps.call_count);

	time(&t);
	tm = localtime(&t);
	strftime(ts, sizeof(ts), "%H:%M:%S", tm);

	printf("%-9s\n", ts);

	key = 0;
	uint64_t value[2000];
	bpf_map_lookup_elem(fd, &key, &value);
	for (uint64_t i = 0; i < thread_count; i++) {
		printf("Thread %lu: %lu\n", i, value[i]);
	}

	fflush(stdout);
	return err;
}

int main(int argc, char **argv)
{
	struct cuda_probe_bpf *skel;
	int err;

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = cuda_probe_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}

	/* Load & verify BPF programs */
	err = cuda_probe_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}
	err = cuda_probe_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton\n");
		goto cleanup;
	}
	while (!exiting) {
		sleep(1);
		print_stat(skel, 7);
	}
cleanup:
	/* Clean up */
	cuda_probe_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
