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
#include "./.output/rocm_probe.skel.h"
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

static int print_stat(struct rocm_probe_bpf *obj)
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
		printf("	calls: %" PRIu64 "\n", value);

		prev_key = &key;
	}
	fflush(stdout);
	return err;
}

int main(int argc, char **argv)
{
	struct rocm_probe_bpf *skel;
	int err;

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = rocm_probe_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}

	/* Load & verify BPF programs */
	err = rocm_probe_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}
	err = rocm_probe_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF skeleton\n");
		goto cleanup;
	}
	while (!exiting) {
		sleep(1);
		print_stat(skel);
	}
cleanup:
	/* Clean up */
	rocm_probe_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
