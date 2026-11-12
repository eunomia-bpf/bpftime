/* SPDX-License-Identifier: MIT */
#include "server_seccomp_hook.hpp"

#if defined(__linux__) && defined(__x86_64__)
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <mutex>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <thread>

#ifndef SECCOMP_RET_USER_NOTIF
#define SECCOMP_RET_USER_NOTIF 0x7fc00000U
#define SECCOMP_FILTER_FLAG_NEW_LISTENER (1UL << 3)
struct seccomp_notif {
	uint64_t id;
	uint32_t pid;
	uint32_t flags;
	struct seccomp_data data;
};
struct seccomp_notif_resp {
	uint64_t id;
	int64_t val;
	int32_t error;
	uint32_t flags;
};
#define SECCOMP_IOCTL_NOTIF_RECV _IOWR('!', 0, struct seccomp_notif)
#define SECCOMP_IOCTL_NOTIF_SEND _IOWR('!', 1, struct seccomp_notif_resp)
#endif

namespace bpftime
{
namespace server
{
namespace
{
std::atomic<int> listener_fd{ -1 };

// The set of syscalls the server needs to see for Go runtimes that bypass the
// libc overrides. Kept small on purpose: the server only emulates resource
// syscalls; everything else is allowed by the filter and never notified.
const int notified_syscalls[] = {
	__NR_bpf,	 __NR_perf_event_open, __NR_ioctl,
	__NR_mmap,	 __NR_munmap,	      __NR_mremap,
	__NR_dup3,	 __NR_fcntl,	      __NR_epoll_create1,
	__NR_epoll_ctl,  __NR_epoll_wait,      __NR_epoll_pwait,
	__NR_close,
};

void worker_loop()
{
	int listener = listener_fd.load(std::memory_order_acquire);
	for (;;) {
		struct seccomp_notif req = {};
		struct seccomp_notif_resp resp = {};
		if (ioctl(listener, SECCOMP_IOCTL_NOTIF_RECV, &req) < 0) {
			if (errno == EINTR || errno == ENOENT)
				continue;
			return;
		}
		const auto &a = req.data.args;
		// Emulate through the server's own dispatcher, which falls
		// back to the real syscall for anything it does not handle.
		int64_t result = (int64_t)::syscall(
			(long)req.data.nr, (long)a[0], (long)a[1], (long)a[2],
			(long)a[3], (long)a[4], (long)a[5]);
		resp.id = req.id;
		if (result < 0 && result >= -4095)
			resp.error = (int32_t)result;
		else
			resp.val = result;
		while (ioctl(listener, SECCOMP_IOCTL_NOTIF_SEND, &resp) < 0 &&
		       errno == EINTR) {
		}
	}
}
} // namespace

bool install_server_seccomp_hook()
{
	if (getenv("BPFTIME_SERVER_SECCOMP") == nullptr)
		return false;
	struct sock_filter filter[2 + std::size(notified_syscalls) * 2 + 1];
	size_t i = 0;
	filter[i++] = (struct sock_filter)BPF_STMT(
		BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch));
	filter[i++] = (struct sock_filter)BPF_JUMP(
		BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0);
	for (int nr : notified_syscalls) {
		filter[i++] = (struct sock_filter)BPF_JUMP(
			BPF_JMP | BPF_JEQ | BPF_K, (uint32_t)nr, 0, 1);
		filter[i++] = (struct sock_filter)BPF_STMT(
			BPF_RET | BPF_K, SECCOMP_RET_USER_NOTIF);
	}
	filter[i++] = (struct sock_filter)BPF_STMT(BPF_RET | BPF_K,
						  SECCOMP_RET_ALLOW);
	struct sock_fprog prog = {
		.len = (unsigned short)i,
		.filter = filter,
	};
	if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0)
		return false;
	int fd = (int)syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER,
			     SECCOMP_FILTER_FLAG_NEW_LISTENER, &prog);
	if (fd < 0)
		return false;
	listener_fd.store(fd, std::memory_order_release);
	std::thread(worker_loop).detach();
	return true;
}
} // namespace server
} // namespace bpftime

#else
namespace bpftime
{
namespace server
{
bool install_server_seccomp_hook()
{
	return false; // seccomp user-notification supported on Linux x86_64
}
} // namespace server
} // namespace bpftime
#endif
