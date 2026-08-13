/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

static int global_shm_exists(void)
{
	const char *shm_name = getenv("BPFTIME_GLOBAL_SHM_NAME");
	if (shm_name == NULL || shm_name[0] == '\0') {
		return 0;
	}

	char path[512];
	int written = snprintf(path, sizeof(path), "/dev/shm/%s", shm_name);
	if (written < 0 || (size_t)written >= sizeof(path)) {
		return 0;
	}
	return access(path, F_OK) == 0;
}

static int fd_is_unlinked_mock_tmp_file(int fd)
{
	char proc_path[64];
	int written = snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d",
			       fd);
	if (written < 0 || (size_t)written >= sizeof(proc_path)) {
		return 0;
	}

	char target[PATH_MAX];
	ssize_t len = readlink(proc_path, target, sizeof(target) - 1);
	if (len < 0) {
		perror("readlink");
		return 0;
	}
	target[len] = '\0';
	const char *prefix = "/tmp/bpftime-mock.";
	return strncmp(target, prefix, strlen(prefix)) == 0 &&
	       strstr(target, " (deleted)") != NULL;
}

static int trigger_startup(void)
{
	/* open() is intentionally not wrapped by handle_exceptions(), so this
	 * verifies that try_startup() itself converts bad_alloc into exit(1).
	 */
	int fd = open("/dev/null", O_RDONLY, 0);
	if (fd < 0) {
		return 101;
	}
	close(fd);
	return 100;
}

static int trigger_startup_ok(void)
{
	int fd = open("/dev/null", O_RDONLY, 0);
	if (fd < 0) {
		return 101;
	}
	close(fd);
	if (getenv("BPFTIME_USED") == NULL || !global_shm_exists()) {
		return 102;
	}
	return 0;
}

static int trigger_startup_wait(void)
{
	int err = trigger_startup_ok();
	if (err != 0) {
		return err;
	}

	const char *ready_file = getenv("BPFTIME_HELPER_READY_FILE");
	const char *release_file = getenv("BPFTIME_HELPER_RELEASE_FILE");
	if (ready_file == NULL || release_file == NULL) {
		return 103;
	}

	FILE *fp = fopen(ready_file, "w");
	if (fp == NULL) {
		perror("fopen ready");
		return 104;
	}
	fclose(fp);

	for (int i = 0; i < 3000; i++) {
		if (access(release_file, F_OK) == 0) {
			return 0;
		}
		usleep(10000);
	}
	return 105;
}

static int trigger_mock_tmp_files(void)
{
	int fd = open("/sys/bus/event_source/devices/uprobe/type",
		      O_RDONLY | O_CLOEXEC, 0);
	if (fd < 0) {
		perror("open");
		return 11;
	}
	if (!fd_is_unlinked_mock_tmp_file(fd)) {
		close(fd);
		return 12;
	}
	if ((fcntl(fd, F_GETFD) & FD_CLOEXEC) == 0) {
		close(fd);
		return 13;
	}
	char buf[64];
	if (read(fd, buf, sizeof(buf)) < 0) {
		perror("read");
		close(fd);
		return 14;
	}
	close(fd);

	FILE *fp = fopen("/sys/bus/event_source/devices/uprobe/format/retprobe",
			 "re");
	if (fp == NULL) {
		perror("fopen");
		return 15;
	}
	fd = fileno(fp);
	if (!fd_is_unlinked_mock_tmp_file(fd)) {
		fclose(fp);
		return 16;
	}
	if ((fcntl(fd, F_GETFD) & FD_CLOEXEC) == 0) {
		fclose(fp);
		return 17;
	}
	if (fgets(buf, sizeof(buf), fp) == NULL && ferror(fp)) {
		perror("fgets");
		fclose(fp);
		return 18;
	}
	fclose(fp);
	return 0;
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
	if (strcmp(argv[1], "startup-ok") == 0) {
		return trigger_startup_ok();
	}
	if (strcmp(argv[1], "startup-wait") == 0) {
		return trigger_startup_wait();
	}
	if (strcmp(argv[1], "mock-tmp") == 0) {
		return trigger_mock_tmp_files();
	}
	if (strcmp(argv[1], "perf-mmap") == 0) {
		return trigger_perf_mmap();
	}
	return 64;
}
