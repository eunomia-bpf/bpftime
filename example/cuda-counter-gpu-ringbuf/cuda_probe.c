// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/* Copyright (c) 2020 Facebook */
#define _GNU_SOURCE
#include <dlfcn.h>
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
#include <time.h>
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
struct data {
	int type;
	union {
		struct {
			uint64_t x, y, z;
		} thread;
		uint64_t time;
	};
};

struct state {
	uint64_t total_time_sum;
	uint64_t count;
};

static bool poll_succeed = false;

static void poll_callback(const void *data, uint64_t size, void *ctx)
{
	poll_succeed = true;
	struct state *state = (struct state *)ctx;
	const struct data *event = data;
	if (event->type == 0) {
		printf("Data from thread: %lu, %lu, %lu\n", event->thread.x,
		       event->thread.y, event->thread.z);
	} else if (event->type == 1) {
		printf("GPU Time usage: %lu\n", event->time);
		state->total_time_sum += event->time;
		state->count += 1;
	}
}

static uint64_t get_timestamp()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
	return ts.tv_sec * (uint64_t)1000000000 + ts.tv_nsec;
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
	int (*poll_fn)(int, void *, void (*)(const void *, uint64_t, void *)) =
		dlsym(RTLD_DEFAULT,
		      "bpftime_syscall_server__poll_gpu_ringbuf_map");
	if (poll_fn == NULL) {
		puts("This example can only be used under bpftime!");
		goto cleanup;
	}
	int mapfd = bpf_map__fd(skel->maps.rb);
	struct state gpu_state, cpu_state;
	while (!exiting) {
		sleep(1);
		poll_succeed = false;
		uint64_t begin = get_timestamp();
		err = poll_fn(mapfd, &gpu_state, poll_callback);
		uint64_t end = get_timestamp();
		if (err < 0) {
			printf("Unable to poll: %d\n", err);
			goto cleanup;
		}
		if (gpu_state.count > 0) {
			printf("Average GPU time usage: %lu (ns), count = %lu\n",
			       gpu_state.total_time_sum / gpu_state.count,
			       gpu_state.count);
		}
		if (poll_succeed) {
			cpu_state.count += 1;
			cpu_state.total_time_sum += end - begin;
		}
		if (cpu_state.count > 0) {
			printf("Average CPU time usage: %lu (ns), count = %lu\n",
			       cpu_state.total_time_sum / cpu_state.count,
			       cpu_state.count);
		}
	}
cleanup:
	/* Clean up */
	cuda_probe_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
