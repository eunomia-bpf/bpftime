/* SPDX-License-Identifier: MIT
 *
 * Seccomp user-notification attach implementation, moved out of the Frida
 * text_segment_transformer. This module is linked into bpftime-agent; it does
 * not link bpftime-syscall-server and reaches the fake kernel only via the
 * get_call_hook() dispatcher. x86_64/Linux only (seccomp user-notif gate).
 */
#if defined(__linux__) && defined(__x86_64__)
#include "seccomp_raw_syscall.hpp"
#include "seccomp_tracer.hpp"
#include "text_segment_transformer.hpp"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include "seccomp_bpf_compat.hpp"
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
#include <unordered_map>

namespace bpftime
{
namespace seccomp
{
// in bpftime-syscall-server; this transformer reaches it through the
// get_call_hook() dispatcher, which falls back to the real syscall for
// anything it does not emulate. No local fd/mmap ledger is kept here.
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

static bool is_tracepoint_id_path(const char *path)
{
	if (path == nullptr)
		return false;
	constexpr const char *debugfs_prefix =
		"/sys/kernel/debug/tracing/events/";
	constexpr const char *tracefs_prefix = "/sys/kernel/tracing/events/";
	size_t length = strlen(path);
	return length >= 3 && strcmp(path + length - 3, "/id") == 0 &&
	       (strncmp(path, debugfs_prefix, strlen(debugfs_prefix)) == 0 ||
		strncmp(path, tracefs_prefix, strlen(tracefs_prefix)) == 0);
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
	// Everything else (bpf/perf_event_open/mmap/epoll/fd lifecycle/...) goes
	// to the syscall server, which owns the emulated resource state and
	// falls back to the real syscall for anything it does not intercept.
	errno = 0;
	int64_t result;
	if (args[0] == __NR_epoll_wait || args[0] == __NR_epoll_pwait) {
		result = bpftime::get_call_hook()(
			args[0], args[1], args[2], args[3], args[4],
			args[5], args[6], 0, 0, 0);
	} else {
		std::lock_guard lock(dispatch_mutex);
		result = bpftime::get_call_hook()(
			args[0], args[1], args[2], args[3], args[4],
			args[5], args[6], 0, 0, 0);
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
					std::memory_order_acquire)) < 0)
				std::this_thread::yield();
			for (;;) {
				struct seccomp_notif request = {};
				struct seccomp_notif_resp response = {};
				if (ioctl(listener, SECCOMP_IOCTL_NOTIF_RECV,
					  &request) < 0) {
					if (errno == EINTR || errno == ENOENT)
						continue;
					return;
				}
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
				while (ioctl(listener, SECCOMP_IOCTL_NOTIF_SEND,
					     &response) < 0 && errno == EINTR) {
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
		BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
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
		BPFTIME_NOTIFY_SYSCALL(__NR_epoll_wait),
		BPFTIME_NOTIFY_SYSCALL(__NR_epoll_pwait),
#undef BPFTIME_NOTIFY_SYSCALL
		BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
	};
	struct sock_fprog program = {
		.len = (unsigned short)(sizeof(filter) / sizeof(filter[0])),
		.filter = filter,
	};
	if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0)
		return false;
	int listener = (int)syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER,
				    SECCOMP_FILTER_FLAG_NEW_LISTENER, &program);
	if (listener < 0)
		return false;
	go_seccomp_listener.store(listener, std::memory_order_release);
	SPDLOG_INFO("Using rootless seccomp notification for Go syscalls");
	return true;
}


bool install_seccomp_tracer()
{
	if (getenv("BPFTIME_GO_SECCOMP") == nullptr)
		return false;
	return setup_go_seccomp_tracer();
}

} // namespace seccomp
} // namespace bpftime
#endif
