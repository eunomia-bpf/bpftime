// SPDX-License-Identifier: GPL-2.0
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include "get_stack_id_example.skel.h"
#include "./get_stack_id_example.h"

#define MAX_STACK_DEPTH 20

static volatile bool exiting = false;

static void sig_handler(int sig)
{
	exiting = true;
}

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

static int handle_event(void *ctx, void *data, size_t data_sz)
{
	const struct event *e = data;
	int i, stack_map_fd = *(int *)ctx;
	uint64_t stack[MAX_STACK_DEPTH] = {};

	const char *func_to_call;
	if (e->operation == MALLOC_ENTER)
		func_to_call = "malloc enter";
	else if (e->operation == FREE_ENTER)
		func_to_call = "free enter";
	else
		func_to_call = "unknown";
	printf("PID: %d, Function: %s", e->pid, func_to_call);

	printf("\n");

	if (e->stack_id >= 0) {
		int err =
			bpf_map_lookup_elem(stack_map_fd, &e->stack_id, stack);
		printf("got err=%d\n",err);
		if (err == 0) {
			printf("Call stack:\n");
			for (i = 0; i < MAX_STACK_DEPTH && stack[i]; i++) {
				printf("  %#lx\n", stack[i]);
			}
		} else {
			printf("Failed to get stack trace for stack_id=%d\n",
			       e->stack_id);
		}
	} else {
		printf("No stack trace available (stack_id=%d)\n", e->stack_id);
	}
	printf("\n");

	return 0;
}

int main(int argc, char **argv)
{
	struct get_stack_id_example_bpf *skel;
	int err, stack_map_fd;
	struct ring_buffer *rb = NULL;

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = get_stack_id_example_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	/* Load BPF programs */
	err = get_stack_id_example_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	/* Attach tracepoints */
	err = get_stack_id_example_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton: %d\n", err);
		goto cleanup;
	}

	/* Get file descriptor for the stack_traces map */
	stack_map_fd = bpf_map__fd(skel->maps.stack_traces);

	/* Set up ring buffer polling */
	rb = ring_buffer__new(bpf_map__fd(skel->maps.rb), handle_event,
			      &stack_map_fd, NULL);
	if (!rb) {
		err = -1;
		fprintf(stderr, "Failed to create ring buffer\n");
		goto cleanup;
	}

	/* Process events */
	printf("Successfully started! Tracing malloc/free in ./victim...\n");
	while (!exiting) {
		err = ring_buffer__poll(rb, 100 /* timeout, ms */);
		if (err == -EINTR) {
			err = 0;
			break;
		}
		if (err < 0) {
			fprintf(stderr, "Error polling ring buffer: %d\n", err);
			break;
		}
	}

cleanup:
	ring_buffer__free(rb);
	get_stack_id_example_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}
