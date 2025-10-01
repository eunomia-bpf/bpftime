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
#include <string.h>
#include "./.output/launchlate.skel.h"
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

static int print_stat(struct launchlate_bpf *obj)
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
	struct launchlate_bpf *skel;
	int err;
	struct timespec ts_mono, ts_real;
	int64_t offset_ns;
	uint32_t key = 0;

	/* Set up libbpf errors and debug info callback */
	libbpf_set_print(libbpf_print_fn);

	/* Cleaner handling of Ctrl-C */
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	/* Load and verify BPF application */
	skel = launchlate_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open and load BPF skeleton\n");
		return 1;
	}

	/* Load & verify BPF programs */
	err = launchlate_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load and verify BPF skeleton\n");
		goto cleanup;
	}

	/* Calibrate clocks: compute offset between CLOCK_REALTIME and CLOCK_MONOTONIC */
	if (clock_gettime(CLOCK_MONOTONIC, &ts_mono) < 0) {
		fprintf(stderr, "Failed to get CLOCK_MONOTONIC: %s\n", strerror(errno));
		goto cleanup;
	}
	if (clock_gettime(CLOCK_REALTIME, &ts_real) < 0) {
		fprintf(stderr, "Failed to get CLOCK_REALTIME: %s\n", strerror(errno));
		goto cleanup;
	}

	/* Calculate offset: realtime - monotonic */
	offset_ns = (int64_t)(ts_real.tv_sec * 1000000000ULL + ts_real.tv_nsec) -
		    (int64_t)(ts_mono.tv_sec * 1000000000ULL + ts_mono.tv_nsec);

	printf("Clock calibration: REALTIME - MONOTONIC = %ld ns\n", offset_ns);
	printf("  MONOTONIC: %ld.%09ld\n", ts_mono.tv_sec, ts_mono.tv_nsec);
	printf("  REALTIME:  %ld.%09ld\n", ts_real.tv_sec, ts_real.tv_nsec);

	/* Store offset in BPF map */
	err = bpf_map_update_elem(bpf_map__fd(skel->maps.clock_offset), &key, &offset_ns, BPF_ANY);
	if (err) {
		fprintf(stderr, "Failed to update clock_offset map: %s\n", strerror(errno));
		goto cleanup;
	}

	err = launchlate_bpf__attach(skel);
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
	launchlate_bpf__destroy(skel);

	return err < 0 ? -err : 0;
}
