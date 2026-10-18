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
#include "redis-patch.skel.h"
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

int main(int argc, char **argv)
{
	struct redis_patch_bpf *skel;
	int err;

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = redis_patch_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}

	/* Load & verify BPF programs */
	err = redis_patch_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}
	// LIBBPF_OPTS(bpf_uprobe_opts, attach_opts, .func_name = "target_func",
	// 	    .retprobe = false, .attach_mode = PROBE_ATTACH_MODE_PERF);
	// struct bpf_link *attach = bpf_program__attach_uprobe_opts(
	// 	skel->progs.do_uprobe_trace, -1, "example/redis-patch/victim", 0,
	// 	&attach_opts);
	// if (!attach) {
	// 	fprintf(stderr, "Failed to attach BPF skeleton\n");
	// 	err = -1;
	// 	goto cleanup;
	// }
	while (!exiting) {
		sleep(1);
	}
cleanup:
	/* Clean up */
	redis_patch_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
