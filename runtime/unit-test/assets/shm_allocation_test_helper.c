/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

static int fail_mkstemp_calls;

int mkstemp(char *template)
{
	if (getenv("BPFTIME_TEST_FAIL_MKSTEMP") != NULL) {
		fail_mkstemp_calls++;
		errno = EACCES;
		return -1;
	}
	return mkostemp(template, 0);
}

static int trigger_startup(void)
{
	/* Trigger lazy interposer initialization through a normal host call. */
	errno = E2BIG;
	int fd = open("/dev/zero", O_RDONLY, 0);
	if (fd < 0) {
		return 101;
	}
	if (errno != E2BIG)
		return 102;
	char byte;
	errno = EOVERFLOW;
	if (read(fd, &byte, 1) != 1)
		return 103;
	if (errno != EOVERFLOW)
		return 104;
	close(fd);
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

static int trigger_mmap_passthrough(void)
{
#if defined(MAP_FIXED_NOREPLACE)
	size_t length = (size_t)getpagesize();
	void *candidate = mmap(NULL, length, PROT_NONE,
			       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (candidate == MAP_FAILED)
		return 30;
	if (munmap(candidate, length) != 0)
		return 31;
	void *mapped =
		mmap(candidate, length, PROT_READ | PROT_WRITE,
		     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
	if (mapped != candidate)
		return 32;
	if (munmap(mapped, length) != 0)
		return 33;
	return 0;
#else
	return 77;
#endif
}

static int trigger_openat_relative(void)
{
	int dirfd =
		open("/sys/bus/event_source/devices", O_RDONLY | O_DIRECTORY);
	if (dirfd < 0)
		return 77;
	void (*enable_mocking)(void) = (void (*)(void))dlsym(
		RTLD_DEFAULT, "bpftime_test_enable_syscall_mocking");
	if (enable_mocking == NULL)
		return 43;
	enable_mocking();
	int fd = openat(dirfd, "uprobe/type", O_RDONLY);
	if (fd < 0)
		return 40;
	if (fail_mkstemp_calls != 1)
		return 41;
	char byte;
	if (read(fd, &byte, 1) != 1)
		return 42;
	close(fd);
	close(dirfd);
	return 0;
}

static int mapping_is_read_exec(void *address)
{
	FILE *maps = fopen("/proc/self/maps", "r");
	if (maps == NULL)
		return -1;
	char line[1024];
	while (fgets(line, sizeof(line), maps) != NULL) {
		unsigned long begin;
		unsigned long end;
		char perms[5];
		if (sscanf(line, "%lx-%lx %4s", &begin, &end, perms) == 3 &&
		    (unsigned long)address >= begin &&
		    (unsigned long)address < end) {
			fclose(maps);
			return perms[0] == 'r' && perms[1] == '-' &&
			       perms[2] == 'x';
		}
	}
	fclose(maps);
	return -1;
}

static int trigger_transformer_fault(void)
{
	size_t length = (size_t)getpagesize();
	void *code = mmap(NULL, length, PROT_READ | PROT_WRITE,
			  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (code == MAP_FAILED)
		return 25;
	((unsigned char *)code)[0] = 0xc3;
	if (mprotect(code, length, PROT_READ | PROT_EXEC) != 0)
		return 26;
	if (mapping_is_read_exec(code) != 1)
		return 27;

	int (*rewrite_with_fault)(void *, size_t, int) =
		(int (*)(void *, size_t, int))dlsym(
			RTLD_DEFAULT, "bpftime_test_rewrite_segment");
	if (rewrite_with_fault == NULL)
		return 28;
	if (rewrite_with_fault(code, length, PROT_READ | PROT_EXEC) != 2)
		return 29;
	if (mapping_is_read_exec(code) != 1)
		return 30;
	if (munmap(code, length) != 0)
		return 31;
	return 0;
}

static int passthrough_stdio(void)
{
	if (getenv("BPFTIME_TEST_TRANSFORMER_FAIL_AFTER_MPROTECT") != NULL) {
		int status = trigger_transformer_fault();
		if (status != 0)
			return status;
	}
	static const char stdout_message[] = "host stdout\n";
	static const char stderr_message[] = "host stderr\n";
	if (write(STDOUT_FILENO, stdout_message, sizeof(stdout_message) - 1) !=
		    (ssize_t)(sizeof(stdout_message) - 1) ||
	    write(STDERR_FILENO, stderr_message, sizeof(stderr_message) - 1) !=
		    (ssize_t)(sizeof(stderr_message) - 1))
		return 24;
	return 23;
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
	if (strcmp(argv[1], "mmap-passthrough") == 0) {
		return trigger_mmap_passthrough();
	}
	if (strcmp(argv[1], "openat-relative") == 0) {
		return trigger_openat_relative();
	}
	if (strcmp(argv[1], "passthrough") == 0) {
		return passthrough_stdio();
	}
	return 64;
}
