/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 *
 * Helper for the rootless seccomp constructor tests. It reports the
 * Seccomp: line of /proc/self/status so the parent test can observe whether
 * a seccomp filter is actually installed in the process.
 *
 * Two build modes:
 *  - BPFTIME_CUSTOM_START: provide our own _start and never call glibc
 *    __libc_start_main, mirroring runtimes such as Go whose rt0 entry takes
 *    over startup. Only raw syscalls are used on this path.
 *  - default: an ordinary glibc main() program.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

#if defined(BPFTIME_CUSTOM_START)

static long
raw_syscall(long nr, long a0, long a1, long a2, long a3, long a4, long a5)
{
	return syscall(nr, a0, a1, a2, a3, a4, a5);
}

static int
raw_open(const char *path)
{
	return (int)raw_syscall(SYS_openat, AT_FDCWD, (long)path, O_RDONLY, 0,
				0, 0);
}

static int
raw_read(int fd, char *buf, int size)
{
	return (int)raw_syscall(SYS_read, fd, (long)buf, size, 0, 0, 0);
}

static void
raw_write(const char *buf, int size)
{
	(void)raw_syscall(SYS_write, STDOUT_FILENO, (long)buf, size, 0, 0, 0);
}

static void
raw_exit(int code)
{
	(void)raw_syscall(SYS_exit_group, code, 0, 0, 0, 0, 0);
}

/* Freestanding-ish entry: the dynamic loader still runs LD_PRELOAD
 * constructors before transferring here, but glibc __libc_start_main is
 * never invoked. */
void
_start(void)
{
	char buf[4096];
	int fd = raw_open("/proc/self/status");
	if (fd < 0)
		raw_exit(3);
	int n = raw_read(fd, buf, (int)sizeof(buf) - 1);
	if (n <= 0)
		raw_exit(4);
	buf[n] = '\0';

	static const char needle[] = "Seccomp:";
	for (int i = 0; i + (int)sizeof(needle) - 1 <= n; i++) {
		if (i != 0 && buf[i - 1] != '\n')
			continue;
		int j = 0;
		while (needle[j] != '\0' && buf[i + j] == needle[j])
			j++;
		if (needle[j] != '\0')
			continue;
		int end = i;
		while (end < n && buf[end] != '\n')
			end++;
		raw_write(buf + i, end - i);
		raw_write("\n", 1);
		raw_exit(0);
	}
	raw_exit(5);
}

#else

#include <stdio.h>
#include <string.h>

int
main(void)
{
	FILE *status = fopen("/proc/self/status", "r");
	if (status == NULL)
		return 3;
	char line[256];
	while (fgets(line, sizeof(line), status) != NULL) {
		if (strncmp(line, "Seccomp:", 8) == 0) {
			fputs(line, stdout);
			return 0;
		}
	}
	return 5;
}

#endif
