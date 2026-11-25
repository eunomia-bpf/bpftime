/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <stddef.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/time.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <limits.h>
#include <signal.h>
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

static int fd_is_cached_synthetic_file(int fd)
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
	/* Synthetic files are materialized once as regular 0400 files in a
	 * private per-process directory and stay linked for the process
	 * lifetime (cleanup runs at exit), so the fd must name a live file
	 * inside that directory rather than an unlinked temp fd.
	 */
	return strstr(target, "/bpftime-synthetic-fs-") != NULL &&
	       strstr(target, " (deleted)") == NULL;
}

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

static int trigger_synthetic_file_caching(void)
{
	int fd = open("/sys/bus/event_source/devices/uprobe/type",
		      O_RDONLY | O_CLOEXEC, 0);
	if (fd < 0) {
		perror("open");
		return 11;
	}
	if (!fd_is_cached_synthetic_file(fd)) {
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
	if (!fd_is_cached_synthetic_file(fd)) {
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

static volatile sig_atomic_t host_signal_seen;
static void host_signal(int sig) { host_signal_seen = sig; }
__attribute__((constructor)) static void install_host_signal(void)
{
	if (getenv("BPFTIME_TEST_HOST_SIGNAL"))
		signal(SIGUSR1, host_signal);
}

extern void bpftime_agent_main(const char *, int *) __attribute__((weak));
static int check_agent_policy(void)
{
	if (getenv("BPFTIME_TEST_FRIDA")) {
		int resident = 0;
		if (!bpftime_agent_main) return 8;
		bpftime_agent_main("", &resident);
	}

	struct sigaction action;
	if (!getenv("BPFTIME_USED") || sigaction(SIGUSR1, NULL, &action) != 0 ||
	    action.sa_handler != (getenv("BPFTIME_TEST_HOST_SIGNAL") ? host_signal : SIG_DFL))
		return 1;
	if (getenv("BPFTIME_TEST_HOST_SIGNAL")) {
		raise(SIGUSR1);
		if (host_signal_seen != SIGUSR1) return 2;
	}
	const char *opt = getenv("BPFTIME_ENABLE_AGENT_IPC");
	int enabled = getenv("BPFTIME_TEST_FRIDA") || (opt && opt[0] && strcmp(opt, "0") != 0);
	DIR *tasks = opendir("/proc/self/task");
	if (!tasks) return 3;
	int threads = 0;
	struct dirent *entry;
	while ((entry = readdir(tasks)))
		if (entry->d_name[0] != '.') ++threads;
	closedir(tasks);
	if (threads != 1 + enabled) return 4;
	int fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (fd < 0) return 5;
	struct timeval timeout = { 2, 0 };
	setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
	struct sockaddr_un addr = { .sun_family = AF_UNIX };
	int len = snprintf(addr.sun_path + 1, sizeof(addr.sun_path) - 1,
			   "bpftime-agent-%d", getpid());
	int connected = connect(fd, (struct sockaddr *)&addr,
				offsetof(struct sockaddr_un, sun_path) + 1 + len) == 0;
	int result = connected != enabled;
	if (enabled && connected) {
		char response[4] = {0};
		if (send(fd, "detach", 6, MSG_NOSIGNAL) != 6) result = 6;
		shutdown(fd, SHUT_WR);
		if (recv(fd, response, 3, MSG_WAITALL) != 3 || strcmp(response, "ok\n")) result = 7;
	}
	close(fd);
	return result;
}

int main(int argc, char **argv)
{
	if (argc != 2) {
		return 64;
	}
	if (strcmp(argv[1], "agent-policy") == 0) return check_agent_policy();
	if (strcmp(argv[1], "startup") == 0) {
		return trigger_startup();
	}
	if (strcmp(argv[1], "startup-ok") == 0) {
		return trigger_startup_ok();
	}
	if (strcmp(argv[1], "startup-wait") == 0) {
		return trigger_startup_wait();
	}
	if (strcmp(argv[1], "synthetic-fs") == 0) {
		return trigger_synthetic_file_caching();
	}
	if (strcmp(argv[1], "perf-mmap") == 0) {
		return trigger_perf_mmap();
	}
	if (strcmp(argv[1], "check-sigusr1") == 0) {
		return check_sigusr1_default();
	}
	return 64;
}
