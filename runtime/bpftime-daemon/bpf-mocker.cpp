// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
// Copyright (c) 2019 Facebook
// Copyright (c) 2020 Netflix
//
// Based on bpf_mocker(8) from BCC by Brendan Gregg and others.
// 14-Feb-2020   Brendan Gregg   Created this.
#include <argp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <linux/perf_event.h>
#include "bpf-mocker-event.h"
#include "bpf-mocker.skel.h"
#include "daemon_config.hpp"
#include "handle_bpf_event.hpp"

/* Tune the buffer size and wakeup rate. These settings cope with roughly
 * 50k opens/sec.
 */
#define PERF_BUFFER_PAGES 64
#define PERF_BUFFER_TIME_MS 10

/* Set the poll timeout when no events occur. This can affect -d accuracy. */
#define PERF_POLL_TIMEOUT_MS 100

#define NSEC_PER_SEC 1000000000ULL

using namespace bpftime;

static volatile sig_atomic_t exiting = 0;

static struct env env = { .uid = INVALID_UID };

const char *argp_program_version = "bpftime-daemon 0.1";
const char *argp_program_bug_address = "https://github.com/eunomia-bpf/bpftime";
const char argp_program_doc[] = "Trace and modify bpf syscalls\n";

static const struct argp_option opts[] = {
	{ "pid", 'p', "PID", 0, "Process ID to trace" },
	{ "uid", 'u', "UID", 0, "User ID to trace" },
	{ "open", 'o', "OPEN", 0, "Show open events" },
	{ "verbose", 'v', NULL, 0, "Verbose debug output" },
	{ "failed", 'x', NULL, 0, "Failed opens only" },
	{},
};

static error_t parse_arg(int key, char *arg, struct argp_state *state)
{
	static int pos_args;
	long int pid, uid;

	switch (key) {
	case 'v':
		env.verbose = true;
		break;
	case 'x':
		env.failed = true;
		break;
	case 'o':
		env.show_open = true;
		break;
	case 'p':
		errno = 0;
		pid = strtol(arg, NULL, 10);
		if (errno || pid <= 0) {
			fprintf(stderr, "Invalid PID: %s\n", arg);
			argp_usage(state);
		}
		env.pid = pid;
		break;
	case 'u':
		errno = 0;
		uid = strtol(arg, NULL, 10);
		if (errno || uid < 0 || uid >= INVALID_UID) {
			fprintf(stderr, "Invalid UID %s\n", arg);
			argp_usage(state);
		}
		env.uid = uid;
		break;
	case ARGP_KEY_ARG:
		if (pos_args++) {
			fprintf(stderr,
				"Unrecognized positional argument: %s\n", arg);
			argp_usage(state);
		}
		errno = 0;
		break;
	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	if (level == LIBBPF_DEBUG && !env.verbose)
		return 0;
	return vfprintf(stderr, format, args);
}

static void sig_int(int signo)
{
	exiting = 1;
}


static int handle_event_rb(void *ctx, void *data, size_t data_sz)
{
	const struct event *e = (const struct event *)data;
	bpf_event_handler handler(env);

	return 0;
}

void handle_lost_events(void *ctx, int cpu, __u64 lost_cnt)
{
	fprintf(stderr, "Lost %llu events on CPU #%d!\n", lost_cnt, cpu);
}

int main(int argc, char **argv)
{
	LIBBPF_OPTS(bpf_object_open_opts, open_opts);
	static const struct argp argp = {
		.options = opts,
		.parser = parse_arg,
		.doc = argp_program_doc,
	};
	// struct perf_buffer *pb = NULL;
	struct ring_buffer *rb = NULL;
	struct bpf_mocker_bpf *obj = NULL;
	int err;

	libbpf_set_print(libbpf_print_fn);
	err = argp_parse(&argp, argc, argv, 0, NULL, NULL);
	if (err)
		return err;

	obj = bpf_mocker_bpf__open();
	if (!obj) {
		fprintf(stderr, "failed to open BPF object\n");
		goto cleanup;
	}

	/* initialize global data (filtering options) */
	obj->rodata->target_pid = env.pid;
	obj->rodata->disable_modify = true;

	err = bpf_mocker_bpf__load(obj);
	if (err) {
		fprintf(stderr, "failed to load BPF object: %d\n", err);
		goto cleanup;
	}

	err = bpf_mocker_bpf__attach(obj);
	if (err) {
		fprintf(stderr, "failed to attach BPF programs\n");
		goto cleanup;
	}
	/* print headers */
	printf("%-6s %-16s %3s %3s ", "PID", "COMM", "FD", "ERR");
	printf("%s", "PATH");
	printf("\n");

	/* Set up ring buffer polling */
	rb = ring_buffer__new(bpf_map__fd(obj->maps.rb), handle_event_rb, NULL,
			      NULL);
	if (!rb) {
		err = -1;
		fprintf(stderr, "Failed to create ring buffer\n");
		goto cleanup;
	}

	if (signal(SIGINT, sig_int) == SIG_ERR) {
		fprintf(stderr, "can't set signal handler: %s\n",
			strerror(errno));
		err = 1;
		goto cleanup;
	}

	/* main: poll */
	while (!exiting) {
		err = ring_buffer__poll(rb, 100 /* timeout, ms */);
		if (err < 0 && err != -EINTR) {
			fprintf(stderr, "error polling perf buffer: %s\n",
				strerror(-err));
			goto cleanup;
		}
		/* reset err to return 0 if exiting */
		err = 0;
	}

cleanup:
	ring_buffer__free(rb);
	bpf_mocker_bpf__destroy(obj);

	return err != 0;
}
