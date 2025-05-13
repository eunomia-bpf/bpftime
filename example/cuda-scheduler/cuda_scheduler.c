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
#include "cuda_scheduler.skel.h"
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

static int print_stat(struct cuda_scheduler_bpf *obj)
{
	time_t t;
	struct tm *tm;
	char ts[16];
	uint32_t key, *prev_key = NULL;
	uint64_t value;
	int err = 0;
	int fd = bpf_map__fd(obj->maps.call_count);

	time(&t);
	tm = localtime(&t);
	strftime(ts, sizeof(ts), "%H:%M:%S", tm);

	printf("%-9s\n", ts);

	while (1) {
		err = bpf_map_get_next_key(fd, prev_key, &key);
		if (err) {
			if (errno == ENOENT) {
				err = 0;
				break;
			}
			warn("bpf_map_get_next_key failed: %s\n",
			     strerror(errno));
			return err;
		}
		err = bpf_map_lookup_elem(fd, &key, &value);
		if (err) {
			warn("bpf_map_lookup_elem failed: %s\n",
			     strerror(errno));
			return err;
		}
		printf("	pid=%-5" PRIu32 " ", key);
		printf("	malloc calls: %" PRIu64 "\n", value);
		err = bpf_map_delete_elem(fd, &key);
		if (err) {
			warn("bpf_map_delete_elem failed: %s\n",
			     strerror(errno));
			return err;
		}
		prev_key = &key;
	}
	fflush(stdout);
	return err;
}

int main(int argc, char **argv)
{
	struct cuda_scheduler_bpf *skel;
	int err;

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = cuda_scheduler_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}

	/* Load & verify BPF programs */
	err = cuda_scheduler_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}
	LIBBPF_OPTS(bpf_uprobe_opts, attach_opts, .func_name = "matMulTiled",
		    .retprobe = false);
	struct bpf_link *attach = bpf_program__attach_uprobe_opts(
		skel->progs.uprobe_matMulTiled, -1, "victim", 0, &attach_opts);
	if (!attach) {
		fprintf(stderr, "Failed to attach BPF skeleton\n");
		err = -1;
		goto cleanup;
	}
	struct bpf_link *attach_cuda = bpf_program__attach_uprobe_opts(
		skel->progs.uprobe_matMulTiled, -1, "victim", 0, &attach_opts);
	if (!attach_cuda) {
		fprintf(stderr, "Failed to attach BPF skeleton (cuda)\n");
		err = -1;
		goto cleanup;
	}
	while (!exiting) {
		sleep(1);
		print_stat(skel);
	}
cleanup:
	/* Clean up */
	cuda_scheduler_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
