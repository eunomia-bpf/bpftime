// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#define _GNU_SOURCE
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <dlfcn.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

#include "./.output/kernel_trace.skel.h"

struct kernel_trace_event {
	unsigned long long block_x, block_y, block_z;
	unsigned long long thread_x, thread_y, thread_z;
	unsigned long long globaltimer;
};

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile bool exiting;

static void sig_handler(int sig)
{
	exiting = true;
}

struct poll_state {
	uint64_t events;
};

static void poll_callback(const void *data, uint64_t size, void *ctx)
{
	const struct kernel_trace_event *event = data;
	struct poll_state *state = ctx;

	printf(
		"[kernel_trace] ts=%llu block=(%llu,%llu,%llu) thread=(%llu,%llu,%llu)\n",
		event->globaltimer, event->block_x, event->block_y,
		event->block_z, event->thread_x, event->thread_y,
		event->thread_z);
	if (state)
		state->events++;
}

int main(int argc, char **argv)
{
	struct kernel_trace_bpf *skel = NULL;
	int err = 0;

	libbpf_set_print(libbpf_print_fn);
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	skel = kernel_trace_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open kernel_trace skeleton\n");
		return 1;
	}

	err = kernel_trace_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load kernel_trace skeleton: %d\n",
			err);
		goto cleanup;
	}

	err = kernel_trace_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach kernel_trace programs: %d\n",
			err);
		goto cleanup;
	}

	int (*poll_fn)(int, void *,
		       void (*)(const void *, uint64_t, void *)) =
		dlsym(RTLD_DEFAULT,
		      "bpftime_syscall_server__poll_gpu_ringbuf_map");
	if (!poll_fn) {
		fprintf(stderr,
			"bpftime_syscall_server__poll_gpu_ringbuf_map is unavailable; run under bpftime runtime\n");
		err = -ENOENT;
		goto cleanup;
	}

	int map_fd = bpf_map__fd(skel->maps.events);
	struct poll_state state = {};

	while (!exiting) {
		int polled = poll_fn(map_fd, &state, poll_callback);
		if (polled < 0) {
			fprintf(stderr, "poll failed: %d\n", polled);
			err = polled;
			break;
		}
		if (state.events) {
			printf("[kernel_trace] total events: %llu\n",
			       (unsigned long long)state.events);
		}
		sleep(1);
	}

cleanup:
	kernel_trace_bpf__destroy(skel);
	return err;
}
