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
#include "seccomp_tracer.hpp"
#include "seccomp_bpf_compat.hpp"
#include "syscall_server_internal_dispatch.hpp"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/audit.h>
#include <linux/filter.h>

#include <linux/seccomp.h>
#include <mutex>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <sys/ioctl.h>
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


namespace bpftime
{
namespace seccomp
{
// Dispatch is serialized for emulated syscalls that touch shared fake-kernel
// state; epoll waits run uncontended because they never mutate that state.
static std::mutex dispatch_mutex;
static const char *kallsyms_path;
static std::atomic<int> go_seccomp_listener{ -1 };


struct mapped_file_resolution {
	std::string logical_path;
	std::string proc_root_path;
};

static bool parse_mapped_file_path(const char *path, pid_t &pid,
				   unsigned long long &start,
				   unsigned long long &end)
{
	if (path == nullptr)
		return false;
	int consumed = 0;
	unsigned parsed_pid = 0;
	if (sscanf(path, "/proc/%u/map_files/%llx-%llx%n", &parsed_pid,
		   &start, &end, &consumed) != 3 || path[consumed] != '\0' ||
	    parsed_pid == 0 || start >= end)
		return false;
	pid = (pid_t)parsed_pid;
	return true;
}

static std::string unescape_proc_path(std::string path)
{
	std::string result;
	result.reserve(path.size());
	for (size_t i = 0; i < path.size(); i++) {
		if (path[i] == '\\' && i + 3 < path.size() &&
		    path[i + 1] >= '0' && path[i + 1] <= '7' &&
		    path[i + 2] >= '0' && path[i + 2] <= '7' &&
		    path[i + 3] >= '0' && path[i + 3] <= '7') {
			result.push_back((char)((path[i + 1] - '0') * 64 +
					        (path[i + 2] - '0') * 8 +
					        path[i + 3] - '0'));
			i += 3;
		} else {
			result.push_back(path[i]);
		}
	}
	return result;
}

static std::optional<mapped_file_resolution>
resolve_mapped_file_path(const char *path)
{
	pid_t pid;
	unsigned long long wanted_start, wanted_end;
	if (!parse_mapped_file_path(path, pid, wanted_start, wanted_end))
		return {};

	std::string maps_path = "/proc/" + std::to_string(pid) + "/maps";
	FILE *maps = fopen(maps_path.c_str(), "re");
	if (maps == nullptr)
		return {};
	char *line = nullptr;
	size_t capacity = 0;
	std::optional<mapped_file_resolution> result;
	while (getline(&line, &capacity, maps) >= 0) {
		unsigned long long start, end, offset, inode;
		unsigned dev_major, dev_minor;
		char permissions[5] = {};
		int path_offset = 0;
		if (sscanf(line, "%llx-%llx %4s %llx %x:%x %llu %n", &start,
			   &end, permissions, &offset, &dev_major, &dev_minor,
			   &inode, &path_offset) != 7 ||
		    start != wanted_start || end != wanted_end)
			continue;
		std::string logical(line + path_offset);
		while (!logical.empty() &&
		       (logical.back() == '\n' || logical.back() == '\r'))
			logical.pop_back();
		logical = unescape_proc_path(logical);
		if (logical.empty() || logical.front() != '/' ||
		    logical.ends_with(" (deleted)"))
			break;
		std::string redirected = "/proc/" + std::to_string(pid) +
					 "/root" + logical;
		struct stat st = {};
		if (stat(redirected.c_str(), &st) != 0 ||
		    (inode != 0 && st.st_ino != inode))
			break;
		result = mapped_file_resolution{ std::move(logical),
						 std::move(redirected) };
		break;
	}
	free(line);
	fclose(maps);
	return result;
}

static bool is_mapped_file_syscall(const int64_t *args)
{
	if (args[0] != __NR_openat && args[0] != __NR_newfstatat &&
	    args[0] != __NR_readlinkat && args[0] != __NR_statx)
		return false;
	pid_t pid;
	unsigned long long start, end;
	return parse_mapped_file_path(
		reinterpret_cast<const char *>(args[2]), pid, start, end);
}

static std::optional<int64_t> dispatch_mapped_file_syscall(const int64_t *args)
{
	if (!is_mapped_file_syscall(args))
		return {};
	auto resolution = resolve_mapped_file_path(
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
		(int64_t)(uintptr_t)resolution->proc_root_path.c_str();
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
	if (args[0] == __NR_openat && args[2] != 0 &&
	    kallsyms_path != nullptr &&
	    strcmp(reinterpret_cast<const char *>(args[2]),
		   "/proc/kallsyms") == 0) {
		int64_t redirected[7];
		std::copy(args, args + 7, redirected);
		redirected[2] = (int64_t)(uintptr_t)kallsyms_path;
		return seccomp::raw_syscall(
			redirected[0], redirected[1], redirected[2],
			redirected[3], redirected[4], redirected[5],
			redirected[6]);
	}
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
	if (getenv("BPFTIME_GO_SECCOMP") == nullptr)
		return false;
	kallsyms_path = getenv("BPFTIME_KALLSYMS_PATH");
	start_go_seccomp_workers();
	struct sock_filter filter[] = {
		BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
			 offsetof(struct seccomp_data, arch)),
		BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, BPFTIME_SECCOMP_AUDIT_ARCH, 1, 0),
		BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
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
		BPFTIME_NOTIFY_SYSCALL(__NR_read),
		BPFTIME_NOTIFY_SYSCALL(__NR_close),
		BPFTIME_NOTIFY_SYSCALL(__NR_ioctl),
		BPFTIME_NOTIFY_SYSCALL(__NR_mmap),
		BPFTIME_NOTIFY_SYSCALL(__NR_munmap),
		BPFTIME_NOTIFY_SYSCALL(__NR_mremap),
		BPFTIME_NOTIFY_SYSCALL(__NR_brk),
		BPFTIME_NOTIFY_SYSCALL(__NR_fcntl),
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
	if (getenv("BPFTIME_GO_SECCOMP") == nullptr)
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
