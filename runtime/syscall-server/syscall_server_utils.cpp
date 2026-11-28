/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpftime_config.hpp"
#include "bpftime_helper_group.hpp"
#include "bpftime_shm_internal.hpp"
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
#include "cuda.h"
#endif
#include "syscall_context.hpp"
#include "synthetic_kallsyms.hpp"
#if __linux__
// Self-contained rootless tracepoint id synthesis. The syscall name->number
// table is generated inside runtime/ (syscall_id_table target); the syscall
// server must not include or link anything under attach/ to read it.
#include "syscall_id_list.h"
#include "platform_utils.hpp"
#endif
#include <cstdint>
#include <cerrno>
#include <cstdlib>
#include <dlfcn.h>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <mutex>
#include <signal.h>
#include <sys/stat.h>
#include <spdlog/spdlog.h>
#include <bpftime_shm.hpp>
#include <string>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <system_error>
#include <unistd.h>
#ifdef ENABLE_BPFTIME_VERIFIER
#include <bpftime-verifier.hpp>
#include <iomanip>
#include <sstream>
#endif
namespace bpftime
{
static std::once_flag g_startup_once;
static std::exception_ptr g_startup_exception;
static std::string g_syscall_server_shm_name;
static pid_t g_syscall_server_shm_owner_pid = -1;
static bool g_syscall_server_owns_shm = false;
using namespace bpftime;
namespace {
// Probe a tracefs event id file through the next (non-interposed) libc
// symbols. This translation unit belongs to the LD_PRELOADed syscall server,
// whose open/openat/fopen handlers call resolve_redirected_path().
// Opening the file with std::ifstream here re-enters those interposed handlers
// and recurses until the stack is exhausted, so the real syscalls must be
// resolved with RTLD_NEXT.
bool read_real_tracepoint_id(const std::string &path_text, uint32_t &out_id)
{
	using openat_fn = int (*)(int, const char *, int, ...);
	using read_fn = ssize_t (*)(int, void *, size_t);
	using close_fn = int (*)(int);
	static openat_fn next_openat =
		(openat_fn)dlsym(RTLD_NEXT, "openat");
	static read_fn next_read = (read_fn)dlsym(RTLD_NEXT, "read");
	static close_fn next_close = (close_fn)dlsym(RTLD_NEXT, "close");
	if (next_openat == nullptr || next_read == nullptr ||
	    next_close == nullptr)
		return false;
	int fd = next_openat(AT_FDCWD, path_text.c_str(),
			     O_RDONLY | O_CLOEXEC);
	if (fd < 0)
		return false;
	char buffer[32];
	size_t filled = 0;
	while (filled < sizeof(buffer) - 1) {
		ssize_t n = next_read(fd, buffer + filled,
				      sizeof(buffer) - 1 - filled);
		if (n > 0) {
			filled += static_cast<size_t>(n);
			continue;
		}
		if (n == 0)
			break;
		if (errno != EINTR)
			break;
	}
	next_close(fd);
	if (filled == 0)
		return false;
	buffer[filled] = '\0';
	char *end = nullptr;
	unsigned long id = strtoul(buffer, &end, 10);
	if (end == buffer || id == 0 || id > UINT32_MAX)
		return false;
	out_id = static_cast<uint32_t>(id);
	return true;
}

// Whether the real /proc/kallsyms hides its addresses, read through
// RTLD_NEXT for the same reason as read_real_tracepoint_id above. Only a
// small sample is needed: one nonzero address means kptr_restrict does not
// apply here, and the real file must then be served unchanged.
bool real_kallsyms_hides_addresses()
{
	using openat_fn = int (*)(int, const char *, int, ...);
	using read_fn = ssize_t (*)(int, void *, size_t);
	using close_fn = int (*)(int);
	static openat_fn next_openat =
		(openat_fn)dlsym(RTLD_NEXT, "openat");
	static read_fn next_read = (read_fn)dlsym(RTLD_NEXT, "read");
	static close_fn next_close = (close_fn)dlsym(RTLD_NEXT, "close");
	if (next_openat == nullptr || next_read == nullptr ||
	    next_close == nullptr)
		return false;
	int fd = next_openat(AT_FDCWD, "/proc/kallsyms", O_RDONLY | O_CLOEXEC);
	if (fd < 0)
		return false;
	char buffer[1024];
	size_t filled = 0;
	while (filled < sizeof(buffer)) {
		ssize_t n = next_read(fd, buffer + filled,
				      sizeof(buffer) - filled);
		if (n > 0) {
			filled += static_cast<size_t>(n);
			continue;
		}
		if (n == 0)
			break;
		if (errno != EINTR)
			break;
	}
	next_close(fd);
	if (filled == 0)
		return false;
	return kallsyms_addresses_hidden(std::string_view(buffer, filled));
}

#if __linux__
// Rootless synthetic tracepoint id base. Must match the encoding consumed on
// the attach side so a synthesized ".../id" resolves to the same event.
constexpr int32_t rootless_synthetic_id_base = 0x40000000;

// Syscall name -> number map built once from the runtime-generated table.
// Local to this translation unit on purpose: reaching into attach/ for this
// mapping would re-introduce the layering inversion the self-contained
// syscall server exists to avoid.
const std::unordered_map<std::string, int32_t> &
rootless_syscall_name_to_nr()
{
	static const auto map = [] {
		std::unordered_map<std::string, int32_t> built;
		std::istringstream ss(table);
		std::string name;
		int32_t nr = 0;
		while (ss >> name >> nr)
			built.emplace(name, nr);
		// Tracepoint spelling vs kernel syscall name.
		if (auto it = built.find("umount2"); it != built.end())
			built.emplace("umount", it->second);
		return built;
	}();
	return map;
}

// Behaviourally identical to the attach-side synthetic tracepoint id
// encoder (same base, prefixes, bounds and return values), reimplemented
// here so the syscall server keeps zero attach/frida dependencies.
// Returns 0 for events that cannot be encoded so the caller falls back to
// the normal tracefs/open path.
int32_t rootless_synthetic_id_for_event(std::string_view event)
{
	if (event == "raw_syscalls/sys_enter")
		return rootless_synthetic_id_base;
	if (event == "raw_syscalls/sys_exit")
		return rootless_synthetic_id_base + 1;
	constexpr std::string_view enter_prefix = "syscalls/sys_enter_";
	constexpr std::string_view exit_prefix = "syscalls/sys_exit_";
	bool enter = event.starts_with(enter_prefix);
	if (!enter && !event.starts_with(exit_prefix))
		return 0;
	auto name = event.substr(
		enter ? enter_prefix.size() : exit_prefix.size());
	const auto &names = rootless_syscall_name_to_nr();
	auto found = names.find(std::string(name));
	if (name.empty() || found == names.end() || found->second < 0 ||
	    found->second >=
		    (INT32_MAX - rootless_synthetic_id_base) / 2 - 1)
		return 0;
	return rootless_synthetic_id_base + 2 * (found->second + 1) +
	       (enter ? 0 : 1);
}
#endif
// Real libc entry points resolved through RTLD_NEXT. Every filesystem
// operation below runs inside the LD_PRELOADed syscall server, whose
// open/openat/fopen handlers call resolve_redirected_path(); going through
// the interposed symbols would re-enter those handlers and recurse.
struct real_libc_fs {
	using openat_fn = int (*)(int, const char *, int, ...);
	using write_fn = ssize_t (*)(int, const void *, size_t);
	using close_fn = int (*)(int);
	using faccessat_fn = int (*)(int, const char *, int, int);
	openat_fn openat = nullptr;
	write_fn write = nullptr;
	close_fn close = nullptr;
	faccessat_fn faccessat = nullptr;
	using unlink_fn = int (*)(const char *);
	unlink_fn unlink = nullptr;

	static const real_libc_fs &instance()
	{
		static const real_libc_fs fs = [] {
			real_libc_fs fs;
			fs.openat = (openat_fn)dlsym(RTLD_NEXT, "openat");
			fs.write = (write_fn)dlsym(RTLD_NEXT, "write");
			fs.close = (close_fn)dlsym(RTLD_NEXT, "close");
			fs.faccessat =
				(faccessat_fn)dlsym(RTLD_NEXT, "faccessat");
			fs.unlink = (unlink_fn)dlsym(RTLD_NEXT, "unlink");
			return fs;
		}();
		return fs;
	}
};

// Per-process private directory backing the cached synthetic files. The
// owner pid makes fork handling explicit: a child process gets its own
// directory on first use and only the owner removes its directory at exit,
// so a forked child never deletes the parent's live cache.
struct synthetic_fs_state {
	std::mutex lock;
	std::string temp_dir;
	pid_t owner_pid = -1;
	bool atexit_registered = false;
	std::unordered_map<std::string, std::string> cached_files;
};

synthetic_fs_state &synthetic_fs()
{
	static synthetic_fs_state state;
	return state;
}

void cleanup_synthetic_fs() noexcept
{
	auto &state = synthetic_fs();
	std::lock_guard guard(state.lock);
	if (state.temp_dir.empty() || state.owner_pid != getpid())
		return;
	std::error_code ec;
	std::filesystem::remove_all(state.temp_dir, ec);
	state.temp_dir.clear();
	state.cached_files.clear();
	state.owner_pid = -1;
}

// Create the private 0700 temp directory. Caller holds synthetic_fs().lock
// and has already decided the current process owns no live directory.
bool create_synthetic_fs_dir_locked()
{
	auto &state = synthetic_fs();
	const char *tmp_root = getenv("TMPDIR");
	if (tmp_root == nullptr || tmp_root[0] == '\0')
		tmp_root = "/tmp";
	std::string templ = std::string(tmp_root) +
			    "/bpftime-synthetic-fs-" +
			    std::to_string(getpid()) + "-XXXXXX";
	std::vector<char> buffer(templ.begin(), templ.end());
	buffer.push_back('\0');
	if (mkdtemp(buffer.data()) == nullptr) {
		SPDLOG_WARN("Unable to create synthetic fs directory: {}",
			    errno);
		return false;
	}
	state.temp_dir = buffer.data();
	state.owner_pid = getpid();
	state.cached_files.clear();
	if (!state.atexit_registered) {
		// atexit handlers are inherited across fork and run in each
		// process; cleanup_synthetic_fs() re-checks ownership.
		std::atexit(cleanup_synthetic_fs);
		state.atexit_registered = true;
	}
	return true;
}

// Flatten a virtual absolute path into a single safe file name component.
std::string flatten_virtual_path(const std::string &virtual_path)
{
	std::string name;
	name.reserve(virtual_path.size());
	for (char ch : virtual_path)
		name.push_back(ch == '/' ? '_' : ch);
	return name;
}

// Write content once into a 0400 file inside the private directory. All
// syscalls use the RTLD_NEXT entries so materialization cannot recurse into
// the interposed open path. Caller holds synthetic_fs().lock.
std::optional<std::string>
materialize_synthetic_file_locked(const std::string &virtual_path,
				  const std::string &content)
{
	const auto &fs = real_libc_fs::instance();
	if (fs.openat == nullptr || fs.write == nullptr ||
	    fs.close == nullptr)
		return {};
	if (synthetic_fs().temp_dir.empty() &&
	    !create_synthetic_fs_dir_locked())
		return {};
	const std::string real_path =
		synthetic_fs().temp_dir + "/" +
		flatten_virtual_path(virtual_path);
	int fd = fs.openat(AT_FDCWD, real_path.c_str(),
			   O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
	if (fd < 0 && errno == EEXIST) {
		// No cache entry points at this name, so the file is a stale
		// remnant of an aborted materialization in this private dir.
		// Drop it and recreate instead of serving truncated content.
		if (fs.unlink == nullptr ||
		    fs.unlink(real_path.c_str()) != 0)
			return {};
		fd = fs.openat(AT_FDCWD, real_path.c_str(),
			       O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
	}
	if (fd < 0) {
		SPDLOG_WARN("Unable to create synthetic file for {}: {}",
			    virtual_path, errno);
		return {};
	}
	const char *data = content.data();
	size_t bytes_left = content.size();
	bool write_ok = true;
	while (bytes_left > 0) {
		ssize_t written = fs.write(fd, data, bytes_left);
		if (written < 0) {
			if (errno == EINTR)
				continue;
			write_ok = false;
			break;
		}
		if (written == 0) {
			write_ok = false;
			break;
		}
		data += written;
		bytes_left -= static_cast<size_t>(written);
	}
	// Read-only: callers must not mutate content shared through the cache.
	if (fchmod(fd, S_IRUSR) != 0)
		SPDLOG_WARN("Unable to chmod synthetic file {}: {}",
			    real_path, errno);
	if (!write_ok) {
		fs.close(fd);
		if (fs.unlink != nullptr)
			fs.unlink(real_path.c_str());
		return {};
	}
	fs.close(fd);
	return real_path;
}

} // namespace
// Why not use string_view? because parse_uint_from_file requires a c-string
static const std::string UPROBE_TYPE_FILE_NAME =
	"/sys/bus/event_source/devices/uprobe/type";
static const std::string URETPROBE_BIT_FILE_NAME =
	"/sys/bus/event_source/devices/uprobe/format/retprobe";
static const std::string KPROBE_TYPE_FILE_NAME =
	"/sys/bus/event_source/devices/kprobe/type";
static const std::string KRETPROBE_BIT_FILE_NAME =
	"/sys/bus/event_source/devices/kprobe/format/retprobe";

static bool pid_is_alive(int pid)
{
	return kill(pid, 0) == 0 || errno == EPERM;
}

static void remove_syscall_server_global_shm() noexcept
{
	if (g_syscall_server_shm_name.empty()) {
		return;
	}
	// Static teardown (__destruct_shm) may already have unmapped the segment
	// before this atexit handler runs; touching the in-segment pid-set mutex
	// would then fault. The liveness flag lives in ordinary .bss and stays
	// readable, so once cleared only a name-based unlink is safe.
	if (!bpftime_global_shm_is_initialized()) {
		if (g_syscall_server_owns_shm &&
		    getpid() == g_syscall_server_shm_owner_pid) {
			try {
				boost::interprocess::shared_memory_object::remove(
					g_syscall_server_shm_name.c_str());
			} catch (...) {
			}
		}
		return;
	}
	if (getpid() != g_syscall_server_shm_owner_pid) {
		try {
			shm_holder.global_shared_memory
				.remove_pid_from_alive_syscall_server_set(
					getpid());
		} catch (...) {
		}
		return;
	}
	if (!g_syscall_server_owns_shm) {
		try {
			shm_holder.global_shared_memory
				.remove_pid_from_alive_syscall_server_set(
					getpid());
		} catch (...) {
		}
		return;
	}
	shm_lifecycle_lock lifecycle_lock(g_syscall_server_shm_name.c_str());
	try {
		shm_holder.global_shared_memory
			.remove_pid_from_alive_syscall_server_set(getpid());

		bool has_alive_server = false;
		bool server_snapshot_ok =
			shm_holder.global_shared_memory
				.iterate_all_pids_in_alive_syscall_server_set(
					[&](int pid) {
						if (has_alive_server) {
							return;
						}
						if (pid_is_alive(pid)) {
							has_alive_server = true;
						}
					});
		if (!server_snapshot_ok) {
			return;
		}
		if (has_alive_server) {
			return;
		}

		bool has_alive_agent = false;
		bool agent_snapshot_ok =
			shm_holder.global_shared_memory
				.iterate_all_pids_in_alive_agent_set([&](int pid) {
					if (has_alive_agent) {
						return;
					}
					if (pid_is_alive(pid)) {
						has_alive_agent = true;
					}
				});
		if (!agent_snapshot_ok) {
			return;
		}
		if (has_alive_agent) {
			return;
		}
		boost::interprocess::shared_memory_object::remove(
			g_syscall_server_shm_name.c_str());
	} catch (...) {
	}
}

static bool initialize_global_shm_with_ownership()
{
	try {
		bpftime_initialize_global_shm(shm_open_type::SHM_CREATE_ONLY);
		return true;
	} catch (const boost::interprocess::interprocess_exception &error) {
		if (error.get_error_code() !=
		    boost::interprocess::already_exists_error) {
			boost::interprocess::shared_memory_object::remove(
				g_syscall_server_shm_name.c_str());
			throw;
		}
	} catch (...) {
		boost::interprocess::shared_memory_object::remove(
			g_syscall_server_shm_name.c_str());
		throw;
	}
	bpftime_initialize_global_shm(shm_open_type::SHM_CREATE_OR_OPEN);
	return false;
}

void start_up(syscall_context &ctx)
{
	std::call_once(g_startup_once, [&ctx]() {
		try {
			SPDLOG_INFO("Starting syscall server..");
			auto runtime_config = construct_runtime_config_from_env();
			SPDLOG_INFO("Initialize syscall server");

			g_syscall_server_shm_name = get_global_shm_name();
			g_syscall_server_shm_owner_pid = getpid();
			shm_lifecycle_lock lifecycle_lock(
				g_syscall_server_shm_name.c_str());
			g_syscall_server_owns_shm =
				initialize_global_shm_with_ownership();
			std::atexit(remove_syscall_server_global_shm);
			if (!shm_holder.global_shared_memory
				     .add_pid_into_alive_syscall_server_set(
					     getpid())) {
				SPDLOG_WARN(
					"Unable to record alive syscall-server pid; disabling automatic shm removal");
				g_syscall_server_owns_shm = false;
			}
#if defined(BPFTIME_ENABLE_CUDA_ATTACH)
			ctx.initialize_cuda();
#endif
			shm_holder.global_shared_memory.begin_new_session();
			shm_holder.global_shared_memory.set_mock_setter(
				[&](bool flg) {
					ctx.enable_mock_after_initialized.store(
						flg, std::memory_order_relaxed);
					SPDLOG_INFO(
						"syscall server: Set enable_mock_after_initialized to {}",
						flg);
				});
#ifdef ENABLE_BPFTIME_VERIFIER
			std::vector<int32_t> helper_ids;
			std::map<int32_t,
				 bpftime::verifier::BpftimeHelperProrotype>
				non_kernel_helpers;
			if (runtime_config.enable_kernel_helper_group) {
				for (auto x : bpftime_helper_group::
					     get_kernel_utils_helper_group()
						     .get_helper_ids()) {
					helper_ids.push_back(x);
				}
			}
			if (runtime_config.enable_shm_maps_helper_group) {
				for (auto x : bpftime_helper_group::
					     get_shm_maps_helper_group()
						     .get_helper_ids()) {
					helper_ids.push_back(x);
				}
			}
			if (runtime_config.enable_ufunc_helper_group) {
				for (auto x : bpftime_helper_group::
					     get_shm_maps_helper_group()
						     .get_helper_ids()) {
					helper_ids.push_back(x);
				}
				// non_kernel_helpers =
				for (const auto &[k, v] :
				     get_ufunc_helper_protos()) {
					non_kernel_helpers[k] = v;
				}
			}
			verifier::set_available_helpers(helper_ids);
			SPDLOG_INFO("Enabling {} helpers", helper_ids.size());
			verifier::set_non_kernel_helpers(non_kernel_helpers);
#endif
			bpftime_set_runtime_config(std::move(runtime_config));
			// Set a variable to indicate the program that it's
			// controlled by bpftime
			setenv("BPFTIME_USED", "1", 0);
			SPDLOG_DEBUG("Set environment variable BPFTIME_USED");
			SPDLOG_INFO("bpftime-syscall-server started");
		} catch (...) {
			g_startup_exception = std::current_exception();
		}
	});
	if (g_startup_exception)
		std::rethrow_exception(g_startup_exception);
}

/*
 * this function is expected to parse integer in the range of [0, 2^31-1] from
 * given file using scanf format string fmt. If actual parsed value is
 * negative, the result might be indistinguishable from error
 */
static int parse_uint_from_file(const char *file, const char *fmt)
{
	int err, ret;
	FILE *f;

	f = fopen(file, "re");
	if (!f) {
		err = -errno;
		SPDLOG_ERROR("Failed to open {}: {}", file, err);
		return err;
	}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
	err = fscanf(f, fmt, &ret);
#pragma GCC diagnostic pop
	if (err != 1) {
		err = err == EOF ? -EIO : -errno;
		SPDLOG_ERROR("Failed to parse {}: {}", file, err);
		fclose(f);
		return err;
	}
	fclose(f);
	return ret;
}

int determine_uprobe_perf_type()
{
	if (!std::filesystem::exists(UPROBE_TYPE_FILE_NAME)) {
		SPDLOG_DEBUG("Using mocked uporbe type value {} for file {}",
			     MOCKED_UPROBE_TYPE_VALUE, UPROBE_TYPE_FILE_NAME);
		return MOCKED_UPROBE_TYPE_VALUE;
	}
	return parse_uint_from_file(UPROBE_TYPE_FILE_NAME.c_str(), "%d\n");
}

int determine_uprobe_retprobe_bit()
{
	if (!std::filesystem::exists(URETPROBE_BIT_FILE_NAME)) {
		SPDLOG_DEBUG("Using mocked uretprobe bit value {} for file {}",
			     MOCKED_URETPROBE_BIT, URETPROBE_BIT_FILE_NAME);
		return MOCKED_URETPROBE_BIT;
	}
	return parse_uint_from_file(URETPROBE_BIT_FILE_NAME.c_str(),
				    "config:%d\n");
}
int determine_kprobe_perf_type()
{
	if (!std::filesystem::exists(KPROBE_TYPE_FILE_NAME)) {
		SPDLOG_DEBUG("Using mocked kprobe type value {} for file {}",
			     MOCKED_KPROBE_TYPE_VALUE, KPROBE_TYPE_FILE_NAME);
		return MOCKED_KPROBE_TYPE_VALUE;
	}
	return parse_uint_from_file(KPROBE_TYPE_FILE_NAME.c_str(), "%d\n");
}
int determine_kprobe_retprobe_bit()
{
	if (!std::filesystem::exists(KRETPROBE_BIT_FILE_NAME)) {
		SPDLOG_DEBUG("Using mocked uretprobe bit value {} for file {}",
			     MOCKED_KRETPROBE_BIT, KRETPROBE_BIT_FILE_NAME);
		return MOCKED_KRETPROBE_BIT;
	}
	return parse_uint_from_file(KRETPROBE_BIT_FILE_NAME.c_str(),
				    "config:%d\n");
}
std::optional<std::string>
get_synthetic_file_content(const std::filesystem::path &path)
{
	auto path_text = path.string();
	// Per-CPU maps keep one column per vCPU slot, so tools size and sum
	// per-CPU values by that count; kernel maps keep the real CPUs.
	if (path_text == "/sys/devices/system/cpu/possible" &&
	    getenv("BPFTIME_RUN_WITH_KERNEL") == nullptr)
		return "0-" + std::to_string(bpftime_vcpu_slots() - 1) + "\n";
	constexpr std::string_view debug_tracepoint_root =
		"/sys/kernel/debug/tracing/events/";
	constexpr std::string_view tracepoint_root =
		"/sys/kernel/tracing/events/";
	const bool is_tracepoint_id =
		(path_text.starts_with(debug_tracepoint_root) ||
		 path_text.starts_with(tracepoint_root)) &&
		path_text.ends_with("/id");
	// Rootless containers expose an all-zero /proc/kallsyms. Serve a
	// built-in synthetic table so an unmodified profiler can initialize
	// without an external System.map, but only while the real file hides
	// its addresses: a privileged kernel reports real ones, and its
	// addresses must survive. An explicit BPFTIME_KALLSYMS_PATH redirect
	// takes precedence in resolve_redirected_path() and never reaches here.
	if (is_synthetic_kallsyms_path(path_text) &&
	    real_kallsyms_hides_addresses()) {
		SPDLOG_DEBUG("serving built-in synthetic /proc/kallsyms");
		return synthetic_kallsyms_content();
	}
	if (is_tracepoint_id) {
		// When tracefs is accessible, the real id file carries a valid
		// kernel tracepoint id. Returning no content lets the normal
		// open path read that ID and keeps kernel-mode opens unchanged.
		uint32_t real_id = 0;
		if (read_real_tracepoint_id(path_text, real_id))
			return {};
		// Encode only syscall events understood by the receiving attach
		// implementation. The same event has one ID under both mounts.
#if __linux__
		const size_t root_size = path_text.starts_with(tracepoint_root) ?
			tracepoint_root.size() : debug_tracepoint_root.size();
		const auto event = std::string_view(path_text).substr(
			root_size, path_text.size() - root_size - 3);
		int32_t id = rootless_synthetic_id_for_event(event);
		if (id != 0)
			return std::to_string(id) + "\n";
		// rootless_synthetic_id_for_event only encodes
		// syscalls/sys_enter_* and syscalls/sys_exit_*. Every other
		// tracepoint (sched/..., raw_syscalls/...) falls back to the
		// same path-derived id used to build the rootless tracepoint
		// table, so the id the profiler reads resolves back to this
		// event when it attaches. Without this fallback an unreadable
		// tracefs id file yields ENOENT and an unmodified profiler
		// aborts during startup.
		return std::to_string(
			       rootless::synthetic_tracepoint_id(path_text)) +
		       "\n";
#endif
		return {};
	}
	if (path == UPROBE_TYPE_FILE_NAME) {
		SPDLOG_DEBUG("{} is uprobe type file", path.c_str());
		return std::to_string(MOCKED_UPROBE_TYPE_VALUE);
	} else if (path == URETPROBE_BIT_FILE_NAME) {
		SPDLOG_DEBUG("{} is uretprobe bit file", path.c_str());
		return "config:" + std::to_string(MOCKED_URETPROBE_BIT);
	} else if (path == KPROBE_TYPE_FILE_NAME) {
		SPDLOG_DEBUG("{} is kprobe type file", path.c_str());
		return std::to_string(MOCKED_KPROBE_TYPE_VALUE);
	} else if (path == KRETPROBE_BIT_FILE_NAME) {
		SPDLOG_DEBUG("{} is kretprobe bit file", path.c_str());
		return "config:" + std::to_string(MOCKED_KRETPROBE_BIT);
	}
	SPDLOG_DEBUG("Unmocked file path: {}", path.c_str());
	return {};
}

std::optional<std::string>
resolve_redirected_path(const std::filesystem::path &full_path)
{
	const std::string virtual_path = full_path.string();
	// Explicit user-supplied overrides redirect straight to the given real
	// file: no generated content, no temp copy, and empty values are treated
	// as unset.
	const char *kallsyms_override = getenv("BPFTIME_KALLSYMS_PATH");
	if (is_synthetic_kallsyms_path(virtual_path) &&
	    kallsyms_override != nullptr && kallsyms_override[0] != '\0')
		return std::string(kallsyms_override);
	const char *btf_override = getenv("BPFTIME_BTF_PATH");
	if (virtual_path == "/sys/kernel/btf/vmlinux" && btf_override != nullptr &&
	    btf_override[0] != '\0')
		return std::string(btf_override);
	// Everything else is only redirected when generated content exists.
	auto content = get_synthetic_file_content(full_path);
	if (!content)
		return {};
	auto &state = synthetic_fs();
	std::lock_guard guard(state.lock);
	// Fork safety: a child inherits the parent's directory path but must not
	// touch it. Drop the inherited mapping and lazily create a directory the
	// child actually owns on its first redirect.
	if (state.owner_pid != -1 && state.owner_pid != getpid()) {
		state.temp_dir.clear();
		state.cached_files.clear();
		state.owner_pid = -1;
	}
	if (auto cached = state.cached_files.find(virtual_path);
	    cached != state.cached_files.end()) {
		const auto &fs = real_libc_fs::instance();
		// Re-materialize if the cached file was removed out of band.
		if (fs.faccessat == nullptr ||
		    fs.faccessat(AT_FDCWD, cached->second.c_str(), F_OK, 0) ==
			    0)
			return cached->second;
		state.cached_files.erase(cached);
	}
	auto real_path =
		materialize_synthetic_file_locked(virtual_path, *content);
	if (!real_path)
		return {};
	state.cached_files.emplace(virtual_path, *real_path);
	return real_path;
}

std::optional<std::filesystem::path>
resolve_filename_and_fd_to_full_path(int fd, const char *file)
{
	if (file == nullptr) {
		return {};
	}
	if (file[0] == '/') {
		return std::filesystem::path(file);
	}
	if (fd == AT_FDCWD) {
		return std::filesystem::path(file);
	}
	std::error_code ec;
	auto dir_path = std::filesystem::read_symlink(
		"/proc/self/fd/" + std::to_string(fd), ec);
	if (dir_path.empty()) {
		SPDLOG_ERROR("Unable to read exact path of fd {}, error={}: ",
			     fd, ec.value(), ec.message());
		return {};
	}
	if (!std::filesystem::is_directory(dir_path)) {
		SPDLOG_ERROR("fd {}, referring {}, is not a directory", fd,
			     dir_path.c_str());
		return {};
	}
	return dir_path / file;
}

} // namespace bpftime
