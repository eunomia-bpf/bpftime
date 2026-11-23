/* SPDX-License-Identifier: MIT
 *
 * Exercises the rootless seccomp USER_NOTIF cBPF argument fast-path.
 *
 * Every syscall under test is issued through a raw x86_64 SYSCALL
 * instruction instead of libc: an LD_PRELOAD interposer would otherwise
 * answer libc calls inside the process before they ever reach the seccomp
 * filter, whereas Go-style raw syscalls (the production rootless workload)
 * traverse seccomp and are either allowed in-kernel or delivered to the
 * bpftime user-notification workers. The process runs with
 * BPFTIME_GO_SECCOMP=1 and bpftime-syscall-server preloaded.
 *
 * Each check prints one "check=<name>=<0|1> errno=<n>" line; the Catch2
 * driver parses them. Exit status is non-zero when any check fails.
 */
#define _GNU_SOURCE
#if defined(__linux__) && defined(__x86_64__)
#include <errno.h>
#include <fcntl.h>
#include <linux/bpf.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

static long
raw_syscall(long nr, long a1, long a2, long a3, long a4, long a5, long a6)
{
	long ret;
	register long r10 __asm__("r10") = a4;
	register long r8 __asm__("r8") = a5;
	register long r9 __asm__("r9") = a6;
	__asm__ volatile("syscall"
			 : "=a"(ret)
			 : "a"(nr), "D"(a1), "S"(a2), "d"(a3), "r"(r10),
			   "r"(r8), "r"(r9)
			 : "rcx", "r11", "memory");
	if ((unsigned long)ret > -4096UL) {
		errno = (int)-ret;
		return -1;
	}
	return ret;
}

static int
bpf_map_create(enum bpf_map_type type, uint32_t key_size,
	       uint32_t value_size, uint32_t max_entries)
{
	union bpf_attr attr;
	memset(&attr, 0, sizeof(attr));
	attr.map_type = type;
	attr.key_size = key_size;
	attr.value_size = value_size;
	attr.max_entries = max_entries;
	return (int)raw_syscall(__NR_bpf, BPF_MAP_CREATE, (long)&attr,
				sizeof(attr), 0, 0, 0);
}

static int
bpf_map_update(int fd, const void *key, const void *value)
{
	union bpf_attr attr;
	memset(&attr, 0, sizeof(attr));
	attr.map_fd = fd;
	attr.key = (long)(uintptr_t)key;
	attr.value = (long)(uintptr_t)value;
	return (int)raw_syscall(__NR_bpf, BPF_MAP_UPDATE_ELEM, (long)&attr,
				sizeof(attr), 0, 0, 0);
}

static int
bpf_map_lookup(int fd, const void *key, void *value)
{
	union bpf_attr attr;
	memset(&attr, 0, sizeof(attr));
	attr.map_fd = fd;
	attr.key = (long)(uintptr_t)key;
	attr.value = (long)(uintptr_t)value;
	return (int)raw_syscall(__NR_bpf, BPF_MAP_LOOKUP_ELEM, (long)&attr,
				sizeof(attr), 0, 0, 0);
}

static int failures = 0;

static void
report(const char *name, int ok, int err)
{
	if (!ok)
		failures++;
	printf("check=%s=%d errno=%d\n", name, ok ? 0 : 1, err);
}

/* Mode 2 means SECCOMP_MODE_FILTER is active for this process. */
static void
report_seccomp_mode(void)
{
	FILE *status = fopen("/proc/self/status", "re");
	if (status == NULL) {
		report("seccomp_mode", 0, errno);
		return;
	}
	char line[256];
	int mode = -1;
	while (fgets(line, sizeof(line), status) != NULL) {
		if (sscanf(line, "Seccomp:\t%d", &mode) == 1)
			break;
	}
	fclose(status);
	report("seccomp_mode", mode == 2, mode < 0 ? ENODATA : mode);
}

int
main(void)
{
	report_seccomp_mode();

	/* Anonymous mmap/munmap: allowed directly in-kernel. */
	{
		void *p = (void *)raw_syscall(
			__NR_mmap, 0, 4096, PROT_READ | PROT_WRITE,
			MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		int ok = p != MAP_FAILED;
		if (ok) {
			memset(p, 0xA5, 4096);
			ok = ((unsigned char *)p)[1023] == 0xA5;
			long r = raw_syscall(__NR_munmap, (long)p, 4096, 0, 0,
					     0, 0);
			ok = ok && r == 0;
		}
		report("anon_mmap_munmap", ok, errno);
	}

	/* mremap is no longer trapped (no emulated handler): kernel handles
	 * it and the relocated contents must survive. */
	{
		void *p = (void *)raw_syscall(
			__NR_mmap, 0, 4096, PROT_READ | PROT_WRITE,
			MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		int ok = p != MAP_FAILED;
		if (ok) {
			((unsigned char *)p)[0] = 0x5A;
			void *q = (void *)raw_syscall(
				__NR_mremap, (long)p, 4096, 8192,
				MREMAP_MAYMOVE, 0, 0);
			ok = q != MAP_FAILED &&
			     ((unsigned char *)q)[0] == 0x5A;
			if (q != MAP_FAILED)
				raw_syscall(__NR_munmap, (long)q, 8192, 0, 0,
					    0, 0);
			else
				raw_syscall(__NR_munmap, (long)p, 4096, 0, 0,
					    0, 0);
		}
		report("mremap", ok, errno);
	}

	/* brk left the filter: a raw brk(0)/brk(same) round trip runs in the
	 * kernel, and ordinary malloc/free must keep working. */
	{
		long current = raw_syscall(__NR_brk, 0, 0, 0, 0, 0, 0);
		long again = raw_syscall(__NR_brk, current, 0, 0, 0, 0, 0);
		report("brk_raw", current > 0 && again == current, errno);
	}
	/* libc malloc/free (brk-based on small arenas, mmap for large ones)
	 * must keep working with the filter installed. */
	{
		void *p = malloc(1UL << 20);
		int ok = p != NULL;
		if (ok) {
			memset(p, 0x3C, 1UL << 20);
			ok = ((unsigned char *)p)[(1UL << 20) - 1] == 0x3C;
		}
		free(p);
		report("brk_malloc", ok, errno);
	}

	int pipefds[2];
	if (pipe(pipefds) != 0) {
		report("pipe_setup", 0, errno);
		return 1;
	}

	/* F_GETFL/F_SETFL: non-DUPFD fcntls run in-kernel. */
	{
		long fl = raw_syscall(__NR_fcntl, pipefds[0], F_GETFL, 0, 0, 0,
				      0);
		int ok = fl >= 0;
		if (ok) {
			long r = raw_syscall(__NR_fcntl, pipefds[0], F_SETFL,
					     fl | O_NONBLOCK, 0, 0, 0);
			ok = r == 0;
			fl = raw_syscall(__NR_fcntl, pipefds[0], F_GETFL, 0,
					 0, 0, 0);
			ok = ok && fl >= 0 && (fl & O_NONBLOCK);
			raw_syscall(__NR_fcntl, pipefds[0], F_SETFL,
				    fl & ~O_NONBLOCK, 0, 0, 0);
		}
		report("fcntl_getfl_setfl", ok, errno);
	}

	/* FIONREAD on a pipe is a non-perf ioctl type: in-kernel. */
	{
		const char msg[7] = "abcdefg";
		ssize_t written = write(pipefds[1], msg, sizeof(msg));
		int ready = -1;
		long r = raw_syscall(__NR_ioctl, pipefds[0], FIONREAD,
				      (long)&ready, 0, 0, 0);
		report("ioctl_fionread",
		       written == (ssize_t)sizeof(msg) && r == 0 &&
			       ready == (int)sizeof(msg),
		       errno);
	}

	/* A BPF map comes from the emulated bpf() trap; its fd is a fake
	 * kernel fd. Successful creation proves the filter is installed and
	 * delivering notifications. */
	int map_fd = bpf_map_create(BPF_MAP_TYPE_HASH, sizeof(uint32_t),
				    sizeof(uint64_t), 8);
	report("bpf_map_create", map_fd >= 0, map_fd < 0 ? errno : 0);
	if (map_fd >= 0) {
		/* F_DUPFD_CLOEXEC must still trap: handle_fcntl registers the
		 * new fd as a bpftime map fd, so map operations on it work. */
		int dup_fd = (int)raw_syscall(__NR_fcntl, map_fd,
					      F_DUPFD_CLOEXEC, 3, 0, 0, 0);
		int cloexec = dup_fd >= 0 &&
			      (fcntl(dup_fd, F_GETFD) & FD_CLOEXEC);
		uint32_t key = 7;
		uint64_t value = 0x1122334455667788ULL;
		uint64_t seen = 0;
		int updated = bpf_map_update(dup_fd, &key, &value);
		int looked_up = bpf_map_lookup(dup_fd, &key, &seen);
		report("fcntl_dupfd_cloexec_map",
		       dup_fd >= 0 && cloexec && updated == 0 &&
			       looked_up == 0 && seen == value,
		       errno);
		if (dup_fd >= 0)
			close(dup_fd);

		/* Ringbuf map fd mmap must trap and return the emulated
		 * consumer page; a kernel mmap on the /dev/null-backed fake
		 * fd would fail with ENODEV. */
		const uint32_t ring_size = 256U * 1024U;
		int rb_fd = bpf_map_create(BPF_MAP_TYPE_RINGBUF, 0, 0,
					   ring_size);
		int mmap_ok = 0;
		if (rb_fd >= 0) {
			void *page = (void *)raw_syscall(
				__NR_mmap, 0, ring_size,
				PROT_READ | PROT_WRITE, MAP_SHARED, rb_fd, 0);
			if (page != MAP_FAILED) {
				mmap_ok = 1;
				raw_syscall(__NR_munmap, (long)page,
					    ring_size, 0, 0, 0, 0);
			}
			close(rb_fd);
		}
		report("mmap_ringbuf_map", mmap_ok, errno);
		close(map_fd);
	}

	/* Software perf event fd: PERF_EVENT_IOC_ENABLE has type '$' (0x24)
	 * and must trap; the emulated enable returns 0 where a kernel ioctl
	 * on the fake fd would return ENOTTY. */
	{
		struct perf_event_attr attr;
		memset(&attr, 0, sizeof(attr));
		attr.type = PERF_TYPE_SOFTWARE;
		attr.size = sizeof(attr);
		attr.config = PERF_COUNT_SW_CPU_CLOCK;
		int perf_fd = (int)raw_syscall(__NR_perf_event_open,
					       (long)&attr, -1, 0, -1, 0, 0);
		int ok = perf_fd >= 0;
		if (ok) {
			long enabled = raw_syscall(__NR_ioctl, perf_fd,
						   PERF_EVENT_IOC_ENABLE, 0, 0,
						   0, 0);
			long disabled = raw_syscall(__NR_ioctl, perf_fd,
						    PERF_EVENT_IOC_DISABLE, 0,
						    0, 0, 0);
			ok = enabled == 0 && disabled == 0;
			close(perf_fd);
		}
		report("ioctl_perf_event_enable", ok, errno);
	}

	close(pipefds[0]);
	close(pipefds[1]);
	printf("SUMMARY failures=%d\n", failures);
	return failures == 0 ? 0 : 1;
}
#else
int
main(void)
{
	printf("SUMMARY skipped=1\n");
	return 0;
}
#endif
