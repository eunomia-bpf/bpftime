/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#include <errno.h>
#include <linux/bpf.h>
#include <linux/perf_event.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

static int trigger_startup(void)
{
	union bpf_attr attr;
	memset(&attr, 0, sizeof(attr));
	attr.map_type = BPF_MAP_TYPE_HASH;
	attr.key_size = sizeof(int);
	attr.value_size = sizeof(int);
	attr.max_entries = 1;
	memcpy(attr.map_name, "allocation_test", sizeof("allocation_test"));

	(void)syscall(__NR_bpf, BPF_MAP_CREATE, &attr, sizeof(attr));
	return 100;
}

static int trigger_perf_mmap(void)
{
	struct perf_event_attr attr;
	memset(&attr, 0, sizeof(attr));
	attr.type = PERF_TYPE_SOFTWARE;
	attr.size = sizeof(attr);
	attr.config = PERF_COUNT_SW_CPU_CLOCK;
	attr.sample_type = PERF_SAMPLE_RAW;

	int fd = syscall(__NR_perf_event_open, &attr, -1, 0, -1, 0);
	if (fd < 0) {
		perror("perf_event_open");
		return 2;
	}

	size_t length = (size_t)getpagesize() + 8 * 1024 * 1024;
	errno = 0;
	void *buffer = mmap(NULL, length, PROT_READ | PROT_WRITE,
			    MAP_SHARED, fd, 0);
	if (buffer != MAP_FAILED || errno != ENOMEM) {
		fprintf(stderr, "mmap=%p errno=%d\n", buffer, errno);
		return 3;
	}
	return 0;
}

int main(int argc, char **argv)
{
	if (argc != 2) {
		return 64;
	}
	if (strcmp(argv[1], "startup") == 0) {
		return trigger_startup();
	}
	if (strcmp(argv[1], "perf-mmap") == 0) {
		return trigger_perf_mmap();
	}
	return 64;
}
