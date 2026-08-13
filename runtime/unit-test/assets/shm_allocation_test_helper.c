/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#include <errno.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

static int trigger_startup(void)
{
	/* The interposer should fall back to the host libc path when bpftime
	 * startup cannot allocate its shared memory.
	 */
	int fd = open("/dev/null", O_RDONLY);
	if (fd < 0) {
		return 101;
	}
	char byte;
	if (read(fd, &byte, 0) != 0) {
		close(fd);
		return 102;
	}
	close(fd);
	fd = syscall(__NR_memfd_create, "bpftime-shm-allocation-test", 0);
	if (fd < 0) {
		return 104;
	}
	close(fd);
	FILE *file = fopen("/dev/null", "r");
	if (file == NULL) {
		return 103;
	}
	fclose(file);
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
	/* The interposer must not leak the caller's stale errno on failure. */
	errno = E2BIG;
	void *buffer =
		mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (buffer != MAP_FAILED || errno != ENOMEM) {
		fprintf(stderr, "mmap=%p errno=%d\n", buffer, errno);
		return 3;
	}
	return 0;
}

static int check_sigusr1_default(void)
{
	struct sigaction action;
	if (sigaction(SIGUSR1, NULL, &action) != 0) {
		return 105;
	}
	return action.sa_handler == SIG_DFL ? 100 : 106;
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
	if (strcmp(argv[1], "check-sigusr1") == 0) {
		return check_sigusr1_default();
	}
	return 64;
}
