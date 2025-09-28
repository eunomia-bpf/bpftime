// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
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
#include <string.h>
#include <sys/syscall.h>
#include <linux/bpf.h>
#include "uprobe_queue.skel.h"

#define warn(...) fprintf(stderr, __VA_ARGS__)

struct event_data {
	uint64_t timestamp;
	uint32_t pid;
	uint32_t tid;
	uint32_t counter;
	uint32_t function_id;
	int32_t input_value;
	char comm[16];
};

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile sig_atomic_t stop;

static void sig_int(int signo)
{
	stop = 1;
}

static long map_pop_elem_syscall(int fd, void *value)
{
	union bpf_attr attr = {};
	attr.map_fd = fd;
	attr.value = (uint64_t)(unsigned long)value;

	return syscall(__NR_bpf, BPF_MAP_LOOKUP_AND_DELETE_ELEM, &attr,
		       sizeof(attr));
}

static long map_peek_elem_syscall(int fd, void *value)
{
	union bpf_attr attr = {};
	attr.map_fd = fd;
	attr.value = (uint64_t)(unsigned long)value;

	return syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, &attr, sizeof(attr));
}

static int process_queue_events(int queue_fd)
{
	struct event_data event;
	int events_processed = 0;

	while (1) {
		int ret = map_pop_elem_syscall(queue_fd, &event);
		if (ret != 0) {
			if (ret == -ENOENT || errno == ENOENT) {
				break;
			} else {
				warn("Failed to pop element from queue: %d (errno: %d)\n",
				     ret, errno);
				return -1;
			}
		}

		events_processed++;

		time_t timestamp_sec = event.timestamp / 1000000000ULL;
		uint64_t timestamp_nsec = event.timestamp % 1000000000ULL;
		struct tm *tm_info = localtime(&timestamp_sec);
		char time_str[64];
		strftime(time_str, sizeof(time_str), "%H:%M:%S", tm_info);

		printf("[%s.%03lu] ", time_str, timestamp_nsec / 1000000);

		if (event.function_id == 1) {
			printf("target_function() called - PID:%u TID:%u input:%d counter:%u process:%s\n",
			       event.pid, event.tid, event.input_value,
			       event.counter, event.comm);
		} else if (event.function_id == 2) {
			printf("secondary_function() called - PID:%u TID:%u counter:%u process:%s\n",
			       event.pid, event.tid, event.counter, event.comm);
		} else {
			printf("unknown function (ID:%u) called - PID:%u TID:%u counter:%u process:%s\n",
			       event.function_id, event.pid, event.tid,
			       event.counter, event.comm);
		}
	}

	return events_processed;
}

static void show_queue_stats(int queue_fd)
{
	struct event_data temp_event;

	int peek_ret = map_peek_elem_syscall(queue_fd, &temp_event);

	if (peek_ret == 0) {
		printf("Queue status: non-empty (head event: function_id=%u, counter=%u)\n",
		       temp_event.function_id, temp_event.counter);
	} else if (peek_ret == -ENOENT || errno == ENOENT) {
		printf("Queue status: empty\n");
	} else {
		printf("Queue status: query failed (error: %d, errno: %d)\n",
		       peek_ret, errno);
	}
}

int main(int argc, char **argv)
{
	struct uprobe_queue_bpf *skel;
	int err;
	int queue_fd;

	libbpf_set_print(libbpf_print_fn);

	skel = uprobe_queue_bpf__open();
	if (!skel) {
		warn("Failed to open BPF skeleton\n");
		return 1;
	}

	err = uprobe_queue_bpf__load(skel);
	if (err) {
		warn("Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	err = uprobe_queue_bpf__attach(skel);
	if (err) {
		warn("Failed to attach BPF program: %d\n", err);
		goto cleanup;
	}

	queue_fd = bpf_map__fd(skel->maps.events_queue);
	if (queue_fd < 0) {
		warn("Failed to get queue map fd\n");
		err = -1;
		goto cleanup;
	}

	printf("Queue Map FD: %d\n", queue_fd);
	printf("eBPF program successfully attached to uprobe\n");
	printf("Starting queue event monitoring...\n");
	printf("Press Ctrl+C to stop\n\n");

	if (signal(SIGINT, sig_int) == SIG_ERR) {
		warn("Cannot set signal handler\n");
		err = 1;
		goto cleanup;
	}

	while (!stop) {
		show_queue_stats(queue_fd);

		int events_count = process_queue_events(queue_fd);
		if (events_count < 0) {
			warn("Error processing queue events\n");
			break;
		}

		if (events_count > 0) {
			printf("Processed %d events this round\n\n",
			       events_count);
		}

		sleep(1);
	}

cleanup:
	uprobe_queue_bpf__destroy(skel);
	printf("\nProgram exiting\n");
	return -err;
}