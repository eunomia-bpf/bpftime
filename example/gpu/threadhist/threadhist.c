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
#include <errno.h>
#include "./.output/threadhist.skel.h"
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

static int print_stat(struct threadhist_bpf *obj, uint64_t thread_count)
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

	printf("%-9s Final thread-exit histogram\n", ts);

	key = 0;
	static uint64_t value[1024 * 1024];
	for (uint64_t i = 0; i < thread_count; i++)
		value[i] = UINT64_MAX;
	err = bpf_map_lookup_elem(fd, &key, &value);
	if (err) {
		warn("bpf_map_lookup_elem failed: %d\n", errno);
		return err;
	}
	uint64_t readback_entries = 0;
	for (uint64_t i = 0; i < thread_count; i++)
		readback_entries += value[i] != UINT64_MAX;
	printf("Configured thread entries: %lu\n", thread_count);
	printf("Readback entries: %lu\n", readback_entries);
	printf("Readback bytes: %lu\n", readback_entries * sizeof(value[0]));
	printf("Readback complete: %d\n", readback_entries == thread_count);
	if (readback_entries != thread_count) {
		warn("incomplete GPU thread histogram readback\n");
		return -EIO;
	}
	uint64_t nonzero_threads = 0;
	uint64_t total_exit_probes = 0;
	for (uint64_t i = 0; i < thread_count; i++) {
		if (value[i] == 0)
			continue;
		printf("Thread %lu: %lu\n", i, value[i]);
		nonzero_threads++;
		total_exit_probes += value[i];
	}
	printf("Nonzero threads: %lu\n", nonzero_threads);
	printf("Total exit probes: %lu\n", total_exit_probes);

	fflush(stdout);
	return err;
}

int main(int argc, char **argv)
{
	struct threadhist_bpf *skel;
	int err;

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = threadhist_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}

	/* Load & verify BPF programs */
	err = threadhist_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}
	err = threadhist_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton\n");
		goto cleanup;
	}
	const char *thread_count_env = getenv("BPFTIME_MAP_GPU_THREAD_COUNT");
	if (thread_count_env == NULL || thread_count_env[0] == '\0') {
		fprintf(stderr,
			"BPFTIME_MAP_GPU_THREAD_COUNT must be set explicitly so the verifier and runtime use one width\n");
		err = -EINVAL;
		goto cleanup;
	}
	errno = 0;
	char *thread_count_end = NULL;
	unsigned long long parsed_thread_count =
		strtoull(thread_count_env, &thread_count_end, 10);
	if (errno != 0 || thread_count_end == thread_count_env ||
	    *thread_count_end != '\0' || parsed_thread_count == 0 ||
	    parsed_thread_count > 1024 * 1024) {
		fprintf(stderr,
			"BPFTIME_MAP_GPU_THREAD_COUNT must be an integer in [1, 1048576]\n");
		err = -EINVAL;
		goto cleanup;
	}
	uint64_t thread_count = (uint64_t)parsed_thread_count;
	while (!exiting)
		sleep(1);
	err = print_stat(skel, thread_count);
cleanup:
	/* Clean up */
	threadhist_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
