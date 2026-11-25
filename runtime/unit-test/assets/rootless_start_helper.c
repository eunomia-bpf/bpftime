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
 *    over startup. Every kernel operation uses a raw syscall instruction
 *    rather than a libc wrapper, again like Go: libc symbols interposed by
 *    an LD_PRELOAD library cannot intercept raw instructions, so this path
 *    exercises the dynamic-loader constructor and the seccomp USER_NOTIF
 *    filter without ever entering glibc startup.
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
	long ret;
	register long r10 __asm__("r10") = a3;
	register long r8 __asm__("r8") = a4;
	register long r9 __asm__("r9") = a5;
	__asm__ volatile("syscall"
			 : "=a"(ret)
			 : "a"(nr), "D"(a0), "S"(a1), "d"(a2), "r"(r10),
			   "r"(r8), "r"(r9)
			 : "rcx", "r11", "memory");
	return ret;
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

/* Freestanding entry: the dynamic loader still runs LD_PRELOAD constructors
 * before transferring here, but glibc __libc_start_main is never invoked. */
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
		_exit(3);
	char line[256];
	while (fgets(line, sizeof(line), status) != NULL) {
		if (strncmp(line, "Seccomp:", 8) == 0) {
			fputs(line, stdout);
			/* Flush, then leave via _exit instead of returning from
			 * main: the preload starts detached seccomp worker
			 * threads that are never joined, and a normal glibc
			 * exit runs global/atexit teardown while those workers
			 * still touch the shared syscall context, a pre-existing
			 * startup-independent shutdown race. The runtimes this
			 * fix targets (e.g. Go) terminate with a raw exit_group
			 * and never run that glibc teardown either. This test
			 * covers constructor/interposer arming, not shutdown. */
			fflush(stdout);
			_exit(0);
		}
	}
	_exit(5);
}

#endif
