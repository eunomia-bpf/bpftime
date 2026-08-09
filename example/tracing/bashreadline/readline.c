// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/* Copyright (c) 2020 Facebook */
#include <signal.h>
#include <stdio.h>
#include <time.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include "readline.skel.h"
#include "readline.h"

#define PERF_BUFFER_PAGES 16
#define PERF_POLL_TIMEOUT_MS 100
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

static void handle_event(void *ctx, int cpu, void *data, __u32 data_size)
{
	struct str_t *e = data;
	struct tm *tm;
	char ts[16];
	time_t t;

	time(&t);
	tm = localtime(&t);
	strftime(ts, sizeof(ts), "%H:%m:%S", tm);

	printf("%-9s %-7d %s\n", ts, e->pid, e->str);
	fflush(stdout);
}

static void handle_lost_events(void *ctx, int cpu, __u64 lost_cnt)
{
	warn("lost %llu events on CPU #%d\n", lost_cnt, cpu);
}

int main(int argc, char **argv)
{
	struct readline_bpf *skel;
	struct perf_buffer *pb = NULL;
	int err;

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = readline_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}

	/* Load & verify BPF programs */
	err = readline_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}
	err = readline_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton handle_exit\n");
		goto cleanup;
	}
	pb = perf_buffer__new(bpf_map__fd(skel->maps.events), PERF_BUFFER_PAGES,
			      handle_event, handle_lost_events, NULL, NULL);
	if (!pb) {
		err = -errno;
		warn("failed to open perf buffer: %d\n", err);
		goto cleanup;
	}

	printf("%-9s %-7s %s\n", "TIME", "PID", "COMMAND");
	while (!exiting) {
		err = perf_buffer__poll(pb, PERF_POLL_TIMEOUT_MS);
		if (err < 0 && err != -EINTR) {
			warn("error polling perf buffer: %s\n", strerror(-err));
			goto cleanup;
		}
		err = 0;
	}
cleanup:
	/* Clean up */
	readline_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
