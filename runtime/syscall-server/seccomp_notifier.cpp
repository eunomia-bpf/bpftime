/* SPDX-License-Identifier: MIT
 *
 * Rootless seccomp user-notification capture, compiled into the
 * bpftime-syscall-server shared object itself.
 *
 * ARCHITECTURE INVARIANT — DO NOT BREAK: this file must stay free of any
 * attach/ dependency. There is no frida, no capstone and no text-segment
 * transformer on the rootless path; intercepted syscalls are emulated by
 * calling dispatch_syscall(), which lives in this same object. The only
 * outward seam is the kernel seccomp USER_NOTIF ABI. See runtime/README.md
 * and attach/README.md. Linux x86_64 and aarch64 (seccomp user-notif gate).
 */
#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))
#include "seccomp_raw_syscall.hpp"
#include "bpftime_config.hpp"
#include "seccomp_tracer.hpp"
#include "seccomp_bpf_compat.hpp"
#include "syscall_server_internal_dispatch.hpp"
#include "syscall_server_utils.hpp"
#include "rootless_map_files.hpp"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/audit.h>
#include <filesystem>
#include <linux/filter.h>

#include <linux/seccomp.h>
#include <mutex>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/vfs.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

// AUDIT_ARCH_* value the kernel reports in seccomp_data.arch for this ABI.
#if defined(__x86_64__)
#define BPFTIME_SECCOMP_AUDIT_ARCH AUDIT_ARCH_X86_64
#elif defined(__aarch64__)
#define BPFTIME_SECCOMP_AUDIT_ARCH AUDIT_ARCH_AARCH64
#else
#error "rootless seccomp USER_NOTIF supports only x86_64 and aarch64"
#endif

#ifndef F_DUPFD_CLOEXEC
#define F_DUPFD_CLOEXEC 1030
#endif

namespace bpftime
{
namespace seccomp
{
// Dispatch is serialized for emulated syscalls that touch shared fake-kernel
// state; epoll waits run uncontended because they never mutate that state.
static std::mutex dispatch_mutex;
static std::atomic<int> go_seccomp_listener{ -1 };


static bool is_mapped_file_syscall(const int64_t *args)
{
	if (args[0] != __NR_openat && args[0] != __NR_newfstatat &&
	    args[0] != __NR_readlinkat && args[0] != __NR_statx)
		return false;
	pid_t pid;
	unsigned long long start, end;
	return rootless::parse_map_files_path(
		reinterpret_cast<const char *>(args[2]), pid, start, end);
}

static std::optional<int64_t> dispatch_mapped_file_syscall(const int64_t *args)
{
	if (!is_mapped_file_syscall(args))
		return {};
	auto resolution = rootless::resolve_map_files_path(
		reinterpret_cast<const char *>(args[2]));
	if (!resolution)
		return -EPERM;
	if (args[0] == __NR_readlinkat) {
		size_t size = std::min((size_t)args[4],
				       resolution->logical_path.size());
		memcpy(reinterpret_cast<void *>(args[3]),
		       resolution->logical_path.data(), size);
		return (int64_t)size;
	}
	int64_t redirected[7];
	std::copy(args, args + 7, redirected);
	redirected[2] =
		(int64_t)(uintptr_t)resolution->accessible_path.c_str();
	return seccomp::raw_syscall(redirected[0], redirected[1], redirected[2],
				 redirected[3], redirected[4], redirected[5],
				 redirected[6]);
}

static bool is_tracefs_mount_path(const char *path)
{
	return path != nullptr &&
	       (strcmp(path, "/sys/kernel/tracing") == 0 ||
		strcmp(path, "/sys/kernel/debug/tracing") == 0);
}

// Rewrite path-based syscalls whose virtual path is backed by a real host
// file (explicit BPFTIME_KALLSYMS_PATH/BPFTIME_BTF_PATH overrides and the
// cached synthetic file system) and run them natively. Every redirected
// path is absolute, so the caller's dirfd is replaced with AT_FDCWD.
static std::optional<int64_t>
dispatch_redirected_path_syscall(const int64_t *args)
{
	switch (args[0]) {
	case __NR_openat:
	case __NR_newfstatat:
	case __NR_statx:
	case __NR_faccessat:
#ifdef __NR_faccessat2
	case __NR_faccessat2:
#endif
		break;
	default:
		return {};
	}
	auto *pathname = reinterpret_cast<const char *>(args[2]);
	if (pathname == nullptr)
		return {};
	auto real_path =
		resolve_redirected_path(std::filesystem::path(pathname));
	if (!real_path)
		return {};
	int64_t redirected[7];
	std::copy(args, args + 7, redirected);
	redirected[1] = AT_FDCWD;
	redirected[2] = reinterpret_cast<int64_t>(real_path->c_str());
	return seccomp::raw_syscall(redirected[0], redirected[1],
				    redirected[2], redirected[3],
				    redirected[4], redirected[5],
				    redirected[6]);
}

static int64_t dispatch_seccomp_syscall(const int64_t *args)
{
	// BPF_TOKEN_CREATE is newer than bpftime's bundled UAPI. Reject it before
	// syscall-server startup so userspace loaders fall back to ordinary FDs.
	if (args[0] == __NR_bpf && args[1] == 36)
		return -EINVAL;
	// Transformer-only handling the syscall server has no knowledge of.
	if (args[0] == __NR_statfs && args[1] != 0 && args[2] != 0 &&
	    is_tracefs_mount_path(
		    reinterpret_cast<const char *>(args[1]))) {
		auto *status = reinterpret_cast<struct statfs *>(args[2]);
		memset(status, 0, sizeof(*status));
		status->f_type = 0x74726163;
		return 0;
	}
	if (auto redirected = dispatch_redirected_path_syscall(args);
	    redirected)
		return *redirected;
	if (auto mapped = dispatch_mapped_file_syscall(args); mapped)
		return *mapped;
	// Everything else (bpf/perf_event_open/mmap/epoll/fd lifecycle/...) is
	// emulated inside this same shared object. No function-pointer seam and
	// no attach/ component are involved.
	errno = 0;
	int64_t result;
#ifdef __NR_epoll_wait
	if (args[0] == __NR_epoll_wait || args[0] == __NR_epoll_pwait) {
#else
	if (args[0] == __NR_epoll_pwait) {
#endif
		result = dispatch_syscall(
			args[0], args[1], args[2], args[3], args[4],
			args[5], args[6]);
	} else {
		std::lock_guard lock(dispatch_mutex);
		result = dispatch_syscall(
			args[0], args[1], args[2], args[3], args[4],
			args[5], args[6]);
	}
	if (result == -1 && errno != 0)
		result = -errno;
	return result;
}

static void start_go_seccomp_workers()
{
	constexpr size_t worker_count = 4;
	for (size_t i = 0; i < worker_count; i++) {
		std::thread([] {
			int listener;
			while ((listener = go_seccomp_listener.load(
					std::memory_order_acquire)) < 0) {
				if (listener == -2)
					return; // installation failed before publishing a listener
				std::this_thread::yield();
			}
			for (;;) {
				struct seccomp_notif request = {};
				struct seccomp_notif_resp response = {};
				bool received = false;
				try {
					if (ioctl(listener, SECCOMP_IOCTL_NOTIF_RECV,
						  &request) < 0) {
						if (errno == EINTR || errno == ENOENT)
							continue;
						return;
					}
					received = true;

					int64_t args[] = {
						(int64_t)request.data.nr,
						(int64_t)request.data.args[0],
						(int64_t)request.data.args[1],
						(int64_t)request.data.args[2],
						(int64_t)request.data.args[3],
						(int64_t)request.data.args[4],
						(int64_t)request.data.args[5],
					};
					response.id = request.id;
					int64_t result = dispatch_seccomp_syscall(args);
					if (result < 0 && result >= -4095)
						response.error = (int32_t)result;
					else
						response.val = result;
					// Retry only on a signal; any other SEND failure must be
					// recorded rather than silently dropping the response and
					// leaving the target blocked on the notification.
					while (ioctl(listener,
						     SECCOMP_IOCTL_NOTIF_SEND,
						     &response) < 0) {
						if (errno != EINTR) {
							SPDLOG_ERROR(
								"seccomp response SEND failed for request {}: {}",
								request.id, errno);
							break;
						}
					}
				} catch (...) {
					// An acquired notification still blocks its host thread.
					// If emulation fails, ask the kernel to run the original
					// syscall instead of abandoning that thread indefinitely.
					if (received) {
						response = {};
						response.id = request.id;
						response.flags = SECCOMP_USER_NOTIF_FLAG_CONTINUE;
						while (ioctl(listener, SECCOMP_IOCTL_NOTIF_SEND,
							     &response) < 0 && errno == EINTR) {
						}
					}
				}
			}
		}).detach();
	}
}

static bool setup_go_seccomp_tracer()
{
	if (!get_agent_bootstrap_config().go_seccomp)
		return false;
	start_go_seccomp_workers();
	struct sock_filter filter[] = {
		BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
			 offsetof(struct seccomp_data, arch)),
		BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, BPFTIME_SECCOMP_AUDIT_ARCH, 1, 0),
		// Fail open for foreign-ABI syscalls (e.g. a 32-bit compat
		// syscall on a 64-bit host). This is an interception filter,
		// not a sandbox: it only diverts a fixed set of native-ABI
		// syscalls to user notification and lets every other syscall
		// run natively through the trailing ALLOW rule. A foreign
		// arch uses a different syscall numbering and is outside
		// that set, so it must be allowed as well. KILL_PROCESS would
		// terminate otherwise valid compat workloads with an
		// uncatchable SIGSYS and break the fail-open contract, while
		// returning an errno would wrongly deny syscalls this filter
		// never intends to police. A separately installed sandbox
		// filter still takes effect, because stacked seccomp
		// filters combine by the strictest action and an ALLOW here
		// cannot override a deny from another filter.
		BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
		BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
			 offsetof(struct seccomp_data, nr)),
#define BPFTIME_NOTIFY_SYSCALL(nr)                                            \
	BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, nr, 0, 1),                     \
		BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_USER_NOTIF)
		BPFTIME_NOTIFY_SYSCALL(__NR_bpf),
		BPFTIME_NOTIFY_SYSCALL(__NR_perf_event_open),
		BPFTIME_NOTIFY_SYSCALL(__NR_openat),
		BPFTIME_NOTIFY_SYSCALL(__NR_statfs),
		BPFTIME_NOTIFY_SYSCALL(__NR_newfstatat),
		BPFTIME_NOTIFY_SYSCALL(__NR_readlinkat),
		BPFTIME_NOTIFY_SYSCALL(__NR_statx),
		BPFTIME_NOTIFY_SYSCALL(__NR_faccessat),
#ifdef __NR_faccessat2
		BPFTIME_NOTIFY_SYSCALL(__NR_faccessat2),
#endif
		BPFTIME_NOTIFY_SYSCALL(__NR_close),
		/*
		 * ioctl: handle_ioctl() emulates only the four perf-event
		 * requests PERF_EVENT_IOC_{ENABLE,DISABLE,SET_BPF,ID}, whose
		 * _IOC_TYPE is '$' (0x24). Trap the whole '$' request family
		 * (its other members simply pass through the emulated
		 * handler); run every other ioctl directly in-kernel. The
		 * low 32 bits of each seccomp_data argument are at
		 * offsetof(args[i]).
		 */
		BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_ioctl, 0, 5),
		BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
			 offsetof(struct seccomp_data, args[1])),
		BPF_STMT(BPF_ALU | BPF_AND | BPF_K, 0xff00U),
		BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0x2400U, 0, 1),
		BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_USER_NOTIF),
		BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
		/*
		 * mmap: only file-backed, non-anonymous mappings need the
		 * fake-kernel fd machinery (ringbuf/array maps and software
		 * perf events). Anonymous mappings and fd == -1 run directly
		 * in the kernel. args[3] is the flags word, args[4] the fd.
		 * BPF_JSET tests the MAP_ANONYMOUS bit directly so the filter
		 * uses only jump opcodes the long-term seccomp validator
		 * accepts (no BPF_JNE dependency).
		 */
		BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_mmap, 0, 6),
		BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
			 offsetof(struct seccomp_data, args[3])),
		BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, MAP_ANONYMOUS, 3, 0),
		BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
			 offsetof(struct seccomp_data, args[4])),
		BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0xffffffffU, 1, 0),
		BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_USER_NOTIF),
		BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
		BPFTIME_NOTIFY_SYSCALL(__NR_munmap),
		/*
		 * fcntl: only F_DUPFD (0) and F_DUPFD_CLOEXEC (1030) have
		 * fake-kernel semantics (bpftime map fd duplication, see
		 * syscall_context::handle_fcntl). All other commands are
		 * passed through by that handler and run directly in-kernel.
		 * args[1] holds the command.
		 */
		BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_fcntl, 0, 5),
		BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
			 offsetof(struct seccomp_data, args[1])),
		BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, F_DUPFD, 2, 0),
		BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, F_DUPFD_CLOEXEC, 1, 0),
		BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
		BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_USER_NOTIF),
		BPFTIME_NOTIFY_SYSCALL(__NR_dup3),
		BPFTIME_NOTIFY_SYSCALL(__NR_epoll_create1),
		BPFTIME_NOTIFY_SYSCALL(__NR_epoll_ctl),
#ifdef __NR_epoll_wait
		BPFTIME_NOTIFY_SYSCALL(__NR_epoll_wait),
#endif
		BPFTIME_NOTIFY_SYSCALL(__NR_epoll_pwait),
#undef BPFTIME_NOTIFY_SYSCALL
		BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
	};
	struct sock_fprog program = {
		.len = (unsigned short)(sizeof(filter) / sizeof(filter[0])),
		.filter = filter,
	};
	if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0) {
		go_seccomp_listener.store(-2, std::memory_order_release);
		return false;
	}
	int listener = (int)syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER,
				    SECCOMP_FILTER_FLAG_NEW_LISTENER, &program);
	if (listener < 0) {
		go_seccomp_listener.store(-2, std::memory_order_release);
		return false;
	}
	go_seccomp_listener.store(listener, std::memory_order_release);
	SPDLOG_INFO("Using rootless seccomp notification for Go syscalls");
	return true;
}


bool install_seccomp_tracer()
{
	if (!get_agent_bootstrap_config().go_seccomp)
		return false;
	try {
		return setup_go_seccomp_tracer();
	} catch (...) {
		// Once installed, the kernel filter still needs its published listener.
		if (go_seccomp_listener.load(std::memory_order_acquire) >= 0)
			return true;
		// A later worker constructor can fail after earlier workers start.
		go_seccomp_listener.store(-2, std::memory_order_release);
		return false;
	}
}

} // namespace seccomp
} // namespace bpftime
#endif
