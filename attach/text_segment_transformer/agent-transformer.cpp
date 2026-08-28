#include "spdlog/cfg/env.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/stdout_sinks.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <frida-gum.h>
#if defined(__linux__) && defined(__x86_64__)
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/vfs.h>
#endif
#include <mutex>
#include <optional>
#include "text_segment_transformer.hpp"
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#if defined(__linux__) && defined(__x86_64__)
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
#ifndef SECCOMP_USER_NOTIF_FLAG_CONTINUE
#define SECCOMP_USER_NOTIF_FLAG_CONTINUE (1UL << 0)
#endif
#endif

using main_func_t = int (*)(int, char **, char **);
using shm_destroy_func_t = void (*)(void);

static main_func_t orig_main_func = nullptr;
static shm_destroy_func_t shm_destroy_func = nullptr;

// Whether syscall server was injected using frida. Defaults to true. If
// __libc_start_main was called, it will be set to false
static bool injected_with_frida = true;
static bool agent_initialized_before_main = false;
extern "C" void bpftime_agent_main(const gchar *data, gboolean *stay_resident);
extern "C" int64_t call_orig_syscall(int64_t sys_nr, int64_t arg1,
				      int64_t arg2, int64_t arg3, int64_t arg4,
				      int64_t arg5, int64_t arg6,
				      int64_t user_ip, int64_t user_sp,
				      int64_t user_bp);

static __attribute__((constructor)) void initialize_go_syscall_tracer()
{
	if (getenv("BPFTIME_GO_SECCOMP") == nullptr)
		return;
	gboolean stay_resident = FALSE;
	bpftime_agent_main("", &stay_resident);
	agent_initialized_before_main = true;
}

#if defined(__linux__) && defined(__x86_64__)
static std::mutex go_dispatch_mutex;
static std::mutex go_resource_mutex;
static std::unordered_set<int> go_mock_fds;
static std::unordered_set<int> go_map_fds;
static std::unordered_set<int> go_program_map_fds;
static std::unordered_set<int> go_mock_read_fds;
static std::unordered_set<int> go_ringbuf_fds;
static std::unordered_map<uintptr_t, size_t> go_mock_mmaps;
static const char *kallsyms_path;
static std::atomic<int> go_seccomp_listener{ -1 };
// SECCOMP_USER_NOTIF_FLAG_CONTINUE was added after user notification itself.
// Cache kernels that reject it so notified syscalls can be emulated by a
// pre-filter worker instead of leaving the requesting thread blocked.
static std::atomic<int> go_seccomp_continue_supported{ -1 };

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
	return call_orig_syscall(redirected[0], redirected[1], redirected[2],
				 redirected[3], redirected[4], redirected[5],
				 redirected[6], 0, 0, 0);
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

static bool should_dispatch_go_syscall(const int64_t *args)
{
	if (args[0] == __NR_bpf || args[0] == __NR_perf_event_open)
		return true;
	if (is_mapped_file_syscall(args))
		return true;
	if (args[0] == __NR_openat && args[2] != 0) {
		auto *path = reinterpret_cast<const char *>(args[2]);
		return is_tracepoint_id_path(path) ||
		       (kallsyms_path != nullptr &&
			strcmp(path, "/proc/kallsyms") == 0);
	}
	if (args[0] == __NR_statfs && args[1] != 0)
		return is_tracefs_mount_path(
			reinterpret_cast<const char *>(args[1]));
	std::lock_guard lock(go_resource_mutex);
	if (args[0] == __NR_epoll_create1)
		return !go_ringbuf_fds.empty();
	if (args[0] == __NR_read)
		return go_mock_read_fds.contains((int)args[1]);
	if (args[0] == __NR_mmap)
		return go_mock_fds.contains((int)args[5]);
	if (args[0] == __NR_fcntl || args[0] == __NR_dup3)
		return go_mock_fds.contains((int)args[1]);
	if (args[0] == __NR_close || args[0] == __NR_ioctl)
		return go_mock_fds.contains((int)args[1]);
	if (args[0] == __NR_epoll_ctl || args[0] == __NR_epoll_wait ||
	    args[0] == __NR_epoll_pwait)
		return go_mock_fds.contains((int)args[1]);
	if (args[0] == __NR_munmap)
		return go_mock_mmaps.contains((uintptr_t)args[1]);
	return false;
}

static bool bpf_command_returns_fd(int64_t cmd)
{
	// Stable values from Linux UAPI enum bpf_cmd. Including linux/bpf.h here
	// would collide with Frida's bpf_insn enum.
	constexpr int64_t map_create = 0;
	constexpr int64_t prog_load = 5;
	constexpr int64_t raw_tracepoint_open = 17;
	constexpr int64_t btf_load = 18;
	constexpr int64_t link_create = 28;
	return cmd == map_create || cmd == prog_load || cmd == btf_load ||
	       cmd == link_create || cmd == raw_tracepoint_open;
}

struct go_bpf_insn {
	uint8_t code;
	uint8_t registers;
	int16_t offset;
	int32_t immediate;
};

struct go_bpf_prog_load_attr {
	uint32_t prog_type;
	uint32_t insn_count;
	uint64_t insns;
};

static void record_go_program_map_references(int64_t attr_address)
{
	const auto *attr = reinterpret_cast<const go_bpf_prog_load_attr *>(
		(uintptr_t)attr_address);
	if (attr == nullptr || attr->insns == 0)
		return;
	const auto *insns = reinterpret_cast<const go_bpf_insn *>(
		(uintptr_t)attr->insns);
	for (uint32_t i = 0; i < attr->insn_count; ++i) {
		const auto &insn = insns[i];
		const uint8_t source_register = insn.registers >> 4;
		if (insn.code == 0x18 &&
		    (source_register == 1 || source_register == 2) &&
		    insn.immediate >= 0)
			go_program_map_fds.insert(insn.immediate);
	}
}

static void record_go_syscall_result(int64_t sysno, int64_t arg1,
				     int64_t arg2, int64_t result)
{
	if (result < 0)
		return;
	std::lock_guard lock(go_resource_mutex);
	if ((sysno == __NR_bpf && bpf_command_returns_fd(arg1)) ||
	    sysno == __NR_perf_event_open || sysno == __NR_epoll_create1) {
		go_mock_fds.insert((int)result);
		if (sysno == __NR_bpf && arg1 == 0)
			go_map_fds.insert((int)result);
		else if (sysno == __NR_bpf && arg1 == 5)
			record_go_program_map_references(arg2);
		if (sysno == __NR_bpf && arg1 == 0 && arg2 != 0 &&
		    *reinterpret_cast<const uint32_t *>(arg2) == 27) {
			go_ringbuf_fds.insert((int)result);
		}
	} else if (sysno == __NR_openat) {
		auto *path = reinterpret_cast<const char *>(arg2);
		if (is_tracepoint_id_path(path)) {
			go_mock_fds.insert((int)result);
			go_mock_read_fds.insert((int)result);
		}
	} else if (sysno == __NR_mmap) {
		go_mock_mmaps.emplace((uintptr_t)result, (size_t)arg2);
	} else if ((sysno == __NR_fcntl &&
		    (arg2 == F_DUPFD || arg2 == F_DUPFD_CLOEXEC)) ||
		   sysno == __NR_dup3) {
		if (go_mock_fds.contains((int)arg1))
			go_mock_fds.insert((int)result);
		if (go_map_fds.contains((int)arg1))
			go_map_fds.insert((int)result);
		if (go_mock_read_fds.contains((int)arg1))
			go_mock_read_fds.insert((int)result);
		if (go_ringbuf_fds.contains((int)arg1))
			go_ringbuf_fds.insert((int)result);
	} else if (sysno == __NR_munmap) {
		go_mock_mmaps.erase((uintptr_t)arg1);
	} else if (sysno == __NR_close) {
		go_mock_read_fds.erase((int)arg1);
		go_ringbuf_fds.erase((int)arg1);
		go_map_fds.erase((int)arg1);
		go_program_map_fds.erase((int)arg1);
		go_mock_fds.erase((int)arg1);
	}
}

static bool defer_go_mock_fd_close(int fd)
{
	std::lock_guard lock(go_resource_mutex);
	if (!go_mock_fds.contains(fd))
		return false;
	if (go_map_fds.contains(fd) && !go_program_map_fds.contains(fd))
		return false;
	// Loaded BPF programs retain map references after the loader closes its
	// descriptor. Keep bpftime's descriptor alive until this process exits so
	// its shared-memory handler id cannot be reused for a different object.
	go_mock_read_fds.erase(fd);
	go_ringbuf_fds.erase(fd);
	go_map_fds.erase(fd);
	go_program_map_fds.erase(fd);
	go_mock_fds.erase(fd);
	return true;
}

static int64_t dispatch_seccomp_syscall(const int64_t *args)
{
	// BPF_TOKEN_CREATE is newer than bpftime's bundled UAPI. Reject it before
	// syscall-server startup so userspace loaders fall back to ordinary FDs.
	if (args[0] == __NR_bpf && args[1] == 36)
		return -EINVAL;
	if (args[0] == __NR_close && defer_go_mock_fd_close((int)args[1]))
		return 0;
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
		return call_orig_syscall(
			redirected[0], redirected[1], redirected[2],
			redirected[3], redirected[4], redirected[5],
			redirected[6], 0, 0, 0);
	}
	if (auto mapped = dispatch_mapped_file_syscall(args); mapped)
		return *mapped;
	int64_t result;
	if (should_dispatch_go_syscall(args)) {
		errno = 0;
		if (args[0] == __NR_epoll_wait ||
		    args[0] == __NR_epoll_pwait) {
			result = bpftime::get_call_hook()(
				args[0], args[1], args[2], args[3], args[4],
				args[5], args[6], 0, 0, 0);
		} else {
			std::lock_guard lock(go_dispatch_mutex);
			result = bpftime::get_call_hook()(
				args[0], args[1], args[2], args[3], args[4],
				args[5], args[6], 0, 0, 0);
		}
		if (result == -1 && errno != 0)
			result = -errno;
	} else {
		result = call_orig_syscall(args[0], args[1], args[2], args[3],
					   args[4], args[5], args[6], 0, 0, 0);
	}
	record_go_syscall_result(args[0], args[1], args[2], result);
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
				if (!should_dispatch_go_syscall(args)) {
					if (go_seccomp_continue_supported.load(
						    std::memory_order_relaxed) != 0) {
						response.flags =
							SECCOMP_USER_NOTIF_FLAG_CONTINUE;
						int send_result;
						do {
							send_result = ioctl(
								listener,
								SECCOMP_IOCTL_NOTIF_SEND,
								&response);
						} while (send_result < 0 &&
							 errno == EINTR);
						if (send_result == 0) {
							go_seccomp_continue_supported.store(
								1,
								std::memory_order_relaxed);
							continue;
						}
						if (errno != EINVAL)
							continue;
						go_seccomp_continue_supported.store(
							0, std::memory_order_relaxed);
						response = {};
						response.id = request.id;
					}
				}
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

#endif

extern "C" int bpftime_hooked_main(int argc, char **argv, char **envp)
{
	if (!agent_initialized_before_main) {
		int stay_resident = 0;
		bpftime_agent_main("", &stay_resident);
	}
	int ret = orig_main_func(argc, argv, envp);
	return ret;
}

extern "C" int __libc_start_main(int (*main)(int, char **, char **), int argc,
				 char **argv,
				 int (*init)(int, char **, char **),
				 void (*fini)(void), void (*rtld_fini)(void),
				 void *stack_end)
{
	injected_with_frida = false;
	orig_main_func = main;
	using this_func_t = decltype(&__libc_start_main);
	this_func_t orig = (this_func_t)dlsym(RTLD_NEXT, "__libc_start_main");

	return orig(bpftime_hooked_main, argc, argv, init, fini, rtld_fini,
		    stack_end);
}

extern "C" void bpftime_agent_main(const gchar *data, gboolean *stay_resident)
{
	auto logger = spdlog::stderr_color_mt("stderr");
	spdlog::set_default_logger(logger);
	spdlog::cfg::load_env_levels();
	/* We don't want to our library to be unloaded after we return. */
	*stay_resident = TRUE;

	const char *agent_so = getenv("AGENT_SO");
	if (agent_so == nullptr) {
		if (std::string(data) != "") {
			SPDLOG_INFO("Using agent path from frida data..");
			agent_so = data;
		} else {
			SPDLOG_ERROR(
				"Please set AGENT_SO to the bpftime-agent when use this tranformer");
			return;
		}
	}
	if (!agent_so) {
		SPDLOG_CRITICAL(
			"Please set AGENT_SO to the bpftime-agent when use this tranformer");
		return;
	}
	SPDLOG_DEBUG("Using agent {}", agent_so);
	cs_arch_register_x86();
	SPDLOG_DEBUG("Loading dynamic library..");
	const bool use_same_namespace =
		getenv("BPFTIME_AGENT_USE_DLOPEN") != nullptr;
	auto next_handle = use_same_namespace ?
				   dlopen(agent_so, RTLD_NOW | RTLD_LOCAL) :
				   dlmopen(LM_ID_NEWLM, agent_so,
					   RTLD_NOW | RTLD_LOCAL);
	if (next_handle == nullptr) {
		SPDLOG_ERROR("Failed to open agent: {}", dlerror());
		return;
	}
	// Set the flag `injected_with_frida` for agent
	bool *injected_with_frida__agent =
		(bool *)dlsym(next_handle, "injected_with_frida");
	if (!injected_with_frida__agent) {
		SPDLOG_WARN(
			"Agent does not expose a symbol named injected_with_frida, so we can't let agent know whether it was loaded using frida");
	} else {
		*injected_with_frida__agent = injected_with_frida;
	}
	auto entry_func = (void (*)(syscall_hooker_func_t *))dlsym(
		next_handle, "_bpftime__setup_syscall_trace_callback");

	if (entry_func) {
		syscall_hooker_func_t orig_syscall_hooker_func =
			bpftime::get_call_hook();
		entry_func(&orig_syscall_hooker_func);
		bpftime::set_call_hook(orig_syscall_hooker_func);
	} else if (auto dispatcher = (syscall_hooker_func_t)dlsym(
			   next_handle,
			   "bpftime_syscall_server__dispatch_raw_syscall");
		   dispatcher) {
		bpftime::set_call_hook(dispatcher);
	} else {
		SPDLOG_CRITICAL(
			"Malformed agent so: no syscall callback entry point");
		return;
	}
#if defined(__linux__) && defined(__x86_64__)
	if (getenv("BPFTIME_GO_SECCOMP") != nullptr) {
		if (!setup_go_seccomp_tracer()) {
			SPDLOG_CRITICAL(
				"Unable to install requested rootless seccomp syscall tracer");
			return;
		}
	} else
#endif
	{
		bpftime::setup_syscall_tracer();
	}
	SPDLOG_DEBUG("Transformer exiting, syscall trace is usable now");
}
