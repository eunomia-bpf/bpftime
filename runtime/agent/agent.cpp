#include "attach_private_data.hpp"
#include "bpf_attach_ctx.hpp"
#include "bpftime_shm_internal.hpp"
#if BPFTIME_HAVE_FRIDA_ATTACH
#include "frida_attach_private_data.hpp"
#include "frida_uprobe_attach_impl.hpp"
#endif
#if BPFTIME_HAVE_TRAP_UPROBE_ATTACH
#include "trap_attach_private_data.hpp"
#include "trap_uprobe_attach_impl.hpp"
#endif
#include "process_filter.hpp"

#include "spdlog/common.h"
#include "bpftime_config.hpp"
#include "bpftime_logger.hpp"
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <random>
#include <string_view>
#include <thread>
#include <unistd.h>
#if BPFTIME_HAVE_FRIDA_ATTACH
#include <frida-gum.h>
#else
typedef char gchar;
typedef int gboolean;
#define TRUE 1
#define FALSE 0
#endif
#include <cstdint>
#include <dlfcn.h>
#include <link.h>
#ifdef __linux__
#include <sys/socket.h>
#include <sys/un.h>
#include "bpftime_agent_ipc.hpp"
#endif
#include "bpftime_shm.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>
#ifdef BPFTIME_ENABLE_CUDA_ATTACH
#include "nv_attach_private_data.hpp"
#include "nv_attach_impl.hpp"
#endif

#if __linux__ && BPFTIME_BUILD_WITH_LIBBPF
#include "syscall_trace_attach_impl.hpp"
#include "syscall_trace_attach_private_data.hpp"
#endif

using namespace bpftime;
using namespace bpftime::attach;
using main_func_t = int (*)(int, char **, char **);

static main_func_t orig_main_func = nullptr;

enum agent_init_state {
	AGENT_UNINITIALIZED = 0,
	AGENT_INITIALIZING = 1,
	AGENT_READY = 2,
};
static int initialized = AGENT_UNINITIALIZED;

using agent_control_fn_t = void (*)(const gchar *);

// Whether this injected process was operated through frida?
// Defaults to true. If __libc_start_main was called, it should be set to false;
// Besides, if agent was loaded by text-transformer, this variable will be set
// by text-transformer
bool injected_with_frida = true;

static std::once_flag agent_ipc_once;
static std::atomic<bool> agent_ipc_stop{ false };
#ifdef __linux__
static int agent_ipc_fd = -1;
static std::thread agent_ipc_thread;
#endif

static std::mutex detach_mutex;
static std::atomic<bool> global_ctx_constructed{ false };

union bpf_attach_ctx_holder {
	bpf_attach_ctx ctx;
	bpf_attach_ctx_holder()
	{
	}
	~bpf_attach_ctx_holder()
	{
	}
	void destroy()
	{
		ctx.~bpf_attach_ctx();
	}
	void init()
	{
		new (&ctx) bpf_attach_ctx;
	}
};
static bpf_attach_ctx_holder ctx_holder;

bpf_attach_ctx &get_global_attach_ctx()
{
	return ctx_holder.ctx;
}

static void apply_injected_kv_overrides(const gchar *data);
static int refresh_attach_session(const gchar *data);
static int perform_detach();

static void install_agent_bootstrap_logger() noexcept
{
	if (__atomic_load_n(&initialized, __ATOMIC_SEQ_CST) !=
	    AGENT_UNINITIALIZED)
		return;
	const char *logger_target = getenv("BPFTIME_LOG_OUTPUT");
	bpftime_set_logger(logger_target == nullptr ? DEFAULT_LOGGER_OUTPUT_PATH :
						     logger_target);
}

static void reset_syscall_trace_global() noexcept
{
#if __linux__ && BPFTIME_BUILD_WITH_LIBBPF
	try {
		// Clear the instance first and keep the original syscall entry
		// published: the text-segment trampoline can still dispatch a
		// syscall after this, and must run the original one then
		// rather than throw or dereference the freed instance.
		bpftime::attach::global_syscall_trace_attach_impl.store(
			nullptr, std::memory_order_release);
	} catch (...) {
	}
#endif
}

extern "C" __attribute__((visibility("default"))) void
bpftime_agent_control(const gchar *data)
{
	injected_with_frida = true;
	// External control entrypoint used to avoid loading multiple agent copies
	// into the same target process (Frida typically loads via /proc/self/fd/*).
	install_agent_bootstrap_logger();
	try {
		(void)refresh_attach_session(data);
	} catch (...) {
	}
}

static void start_agent_ipc_server_once();

static agent_control_fn_t find_other_loaded_agent_control()
{
	if (auto sym = (agent_control_fn_t)dlsym(RTLD_NEXT, "bpftime_agent_control");
	    sym && sym != &bpftime_agent_control) {
		return sym;
	}

	Dl_info self_info {};
	const char *self_name = nullptr;
	if (dladdr((void *)&bpftime_agent_control, &self_info) != 0) {
		self_name = self_info.dli_fname;
	}
	struct FindCtx {
		const char *self_name;
		agent_control_fn_t found;
	} ctx{ self_name, nullptr };

	dl_iterate_phdr(
		[](struct dl_phdr_info *info, size_t, void *data) -> int {
			auto *ctx = static_cast<FindCtx *>(data);
			if (!info || !info->dlpi_name || info->dlpi_name[0] == '\0')
				return 0;
			if (ctx->self_name &&
			    strcmp(info->dlpi_name, ctx->self_name) == 0)
				return 0;
			void *h = dlopen(info->dlpi_name, RTLD_NOLOAD | RTLD_NOW);
			if (!h)
				return 0;
			auto sym = (agent_control_fn_t)dlsym(
				h, "bpftime_agent_control");
			if (sym && sym != &bpftime_agent_control) {
				ctx->found = sym;
				dlclose(h);
				return 1;
			}
			dlclose(h);
			return 0;
		},
		&ctx);

	return ctx.found;
}

#ifdef __linux__
static bool try_forward_to_existing_agent(const gchar *data)
{
	std::string req("refresh ");
	if (data != nullptr)
		req += data;
	return bpftime::try_send_agent_ipc(getpid(), req);
}
#endif

static void start_agent_ipc_server_once()
{
#ifdef __linux__
	const char *enable_ipc = getenv("BPFTIME_ENABLE_AGENT_IPC");
	if (!injected_with_frida &&
	    (enable_ipc == nullptr || enable_ipc[0] == '\0' ||
	     strcmp(enable_ipc, "0") == 0))
		return;
	std::call_once(agent_ipc_once, []() {
		sockaddr_un addr {};
		socklen_t len = 0;
		if (!bpftime::make_agent_ipc_addr(getpid(), addr, len)) {
			SPDLOG_WARN("agent ipc: unable to build socket address");
			return;
		}
		int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
		if (fd < 0) {
			SPDLOG_WARN("agent ipc: socket() failed: {}", strerror(errno));
			return;
		}
		if (::bind(fd, (sockaddr *)&addr, len) != 0) {
			SPDLOG_WARN("agent ipc: bind() failed: {}", strerror(errno));
			::close(fd);
			return;
		}
		if (::listen(fd, 8) != 0) {
			SPDLOG_WARN("agent ipc: listen() failed: {}", strerror(errno));
			::close(fd);
			return;
		}
		agent_ipc_fd = fd;
		agent_ipc_stop.store(false, std::memory_order_release);
		try {
			agent_ipc_thread = std::thread([]() {
			for (;;) {
				if (agent_ipc_stop.load(std::memory_order_acquire))
					return;
				int cfd = ::accept4(agent_ipc_fd, nullptr, nullptr,
						    SOCK_CLOEXEC);
				if (cfd < 0) {
					if (errno == EINTR)
						continue;
					usleep(10 * 1000);
					continue;
				}
				// Basic auth: allow same uid or root.
				struct ucred cred {};
				socklen_t clen = sizeof(cred);
				if (getsockopt(cfd, SOL_SOCKET, SO_PEERCRED, &cred,
					       &clen) != 0) {
					::close(cfd);
					continue;
				}
				if (cred.uid != 0 && (uid_t)cred.uid != geteuid()) {
					::close(cfd);
					continue;
				}
				try {
					bpftime::set_ipc_socket_timeout(cfd, 1000);
					std::string req;
					char buf[4096];
					for (;;) {
						ssize_t n = ::recv(cfd, buf,
								   sizeof(buf), 0);
						if (n <= 0)
							break;
						req.append(buf, buf + n);
						if (req.size() > (1 << 20))
							break;
					}
					// Parse: "refresh <k=v;...>" or "detach" or "status".
					auto trim = [](std::string &s) {
						while (!s.empty() &&
						       (s.back() == '\n' ||
							s.back() == '\r' ||
							s.back() == '\0'))
							s.pop_back();
					};
					trim(req);
					if (req.rfind("refresh ", 0) == 0) {
						std::string arg = req.substr(
							strlen("refresh "));
						int rc = refresh_attach_session(
							arg.c_str());
						if (rc == 0) {
							(void)::send(cfd, "ok\n",
								     3,
								     MSG_NOSIGNAL);
						} else {
							std::string out =
								"err refresh " +
								std::to_string(rc) +
								"\n";
							(void)::send(cfd,
								     out.data(),
								     out.size(),
								     MSG_NOSIGNAL);
						}
					} else if (req == "detach") {
						int rc = perform_detach();
						if (rc == 0) {
							(void)::send(cfd, "ok\n",
								     3,
								     MSG_NOSIGNAL);
						} else {
							std::string out =
								"err detach " +
								std::to_string(rc) +
								"\n";
							(void)::send(cfd,
								     out.data(),
								     out.size(),
								     MSG_NOSIGNAL);
						}
					} else if (req == "status") {
						uint64_t epoch =
							shm_holder
								.global_shared_memory
								.read_stable_epoch_seq();
						std::string out;
						if (epoch ==
						    BPFTIME_EPOCH_SEQ_MISSING) {
							out = "epoch_seq=missing\n";
						} else if (
							epoch ==
							BPFTIME_EPOCH_SEQ_UNSTABLE) {
							out = "epoch_seq=unstable\n";
						} else {
							out = "epoch_seq=" +
							      std::to_string(
								      epoch) +
							      "\n";
						}
						(void)::send(cfd, out.data(),
							     out.size(),
							     MSG_NOSIGNAL);
					} else {
						(void)::send(cfd, "unknown\n", 8,
							     MSG_NOSIGNAL);
					}
				} catch (...) {
					// Never let an attach/IPC exception escape a
					// std::thread in an LD_PRELOAD library and
					// terminate the host process.
					(void)::send(cfd, "err exception\n", 14,
						     MSG_NOSIGNAL);
				}
				::close(cfd);
			}
			});
		} catch (...) {
			agent_ipc_stop.store(true, std::memory_order_release);
			::close(fd);
			agent_ipc_fd = -1;
			return;
		}
		std::atexit([]() {
			agent_ipc_stop.store(true, std::memory_order_release);
			if (agent_ipc_fd >= 0) {
				::close(agent_ipc_fd);
				agent_ipc_fd = -1;
			}
			if (agent_ipc_thread.joinable())
				agent_ipc_thread.detach();
		});
		try {
			SPDLOG_INFO("agent ipc: listening (abstract) for pid {}",
				    (int)getpid());
		} catch (...) {
		}
	});
#else
	(void)0;
#endif
}

static int perform_detach()
{
	std::lock_guard<std::mutex> detach_guard(detach_mutex);
	attach_callback_suppression_scope suppress_callbacks;
	if (!global_ctx_constructed.load(std::memory_order_acquire)) {
		__atomic_store_n(&initialized, AGENT_UNINITIALIZED,
				 __ATOMIC_SEQ_CST);
		return 0;
	}
	SPDLOG_INFO("Detaching..");

	// Detach all instantiated links so GPU code stops using loader-owned
	// resources (CUDA IPC buffers) before the loader exits.
	int detach_err = ctx_holder.ctx.destroy_all_attach_links();
	if (detach_err < 0) {
		SPDLOG_ERROR("Unable to detach cleanly: {}", detach_err);
	}
	// Allow re-instantiating handlers on the next session without requiring
	// destroying the whole attach context (which may block inside driver
	// teardown while kernels are still running).
	ctx_holder.ctx.reset_instantiated_state();
	if (detach_err >= 0) {
		shm_holder.global_shared_memory.remove_pid_from_alive_agent_set(
			getpid());
	}
	SPDLOG_DEBUG("Detaching done");
	bpftime_logger_flush();
	return detach_err < 0 ? detach_err : 0;
}

static void stop_syscall_dispatch_at_exit() noexcept
{
#if __linux__ && BPFTIME_BUILD_WITH_LIBBPF
	try {
		auto *impl =
			bpftime::attach::global_syscall_trace_attach_impl.load(
				std::memory_order_acquire);
		if (impl != nullptr)
			impl->dispatch_process_exit();
	} catch (...) {
	}
#endif
	// libc and DSO teardown may issue mmap/munmap after the runtime shared
	// memory has been destroyed. Stop dispatching BPF programs before those
	// destructors run so their map helpers cannot access an unmapped manager.
	reset_syscall_trace_global();
}

syscall_hooker_func_t orig_hooker;

static void bpftime_agent_main_impl(const gchar *data,
				    gboolean *stay_resident);
extern "C" void bpftime_agent_main(const gchar *data, gboolean *stay_resident);

static void detach_agent_at_process_exit()
{
	(void)perform_detach();
}

static void initialize_preloaded_agent()
{
	int stay_resident = 0;
	injected_with_frida = false;
	bpftime_agent_main_impl("", &stay_resident);
	if (std::atexit(detach_agent_at_process_exit) != 0)
		SPDLOG_WARN("Unable to register bpftime process-exit cleanup");
}

extern "C" int bpftime_hooked_main(int argc, char **argv, char **envp)
{
	initialize_preloaded_agent();
	return orig_main_func(argc, argv, envp);
}

extern "C" int __libc_start_main(int (*main)(int, char **, char **), int argc,
				 char **argv,
				 int (*init)(int, char **, char **),
				 void (*fini)(void), void (*rtld_fini)(void),
				 void *stack_end)
{
	orig_main_func = main;
	using this_func_t = decltype(&__libc_start_main);
	this_func_t orig = (this_func_t)dlsym(RTLD_NEXT, "__libc_start_main");
	// musl sets its program-name symbol only in the libc entry below; use argv[0].
	const char *argv0 = (argc > 0 && argv != nullptr) ? argv[0] : nullptr;
	if (argv0 != nullptr) {
		const char *slash = strrchr(argv0, '/');
		argv0 = slash != nullptr ? slash + 1 : argv0;
	}
	if (agent::should_skip_agent_process(argv0))
		return orig(main, argc, argv, init, fini, rtld_fini, stack_end);
	return orig(bpftime_hooked_main, argc, argv, init, fini, rtld_fini,
		    stack_end);
}
// Simple "k=v" pairs separated by ';', space or tab. Each pair is split at
// the first '=' and passed to the callback in original order; a callback
// returning false stops the iteration.
template <typename Fn>
static void for_each_kv(const gchar *data, Fn &&fn)
{
	if (data == nullptr || data[0] == '\0')
		return;
	std::string_view sv(data);
	for (size_t pos = 0; pos < sv.size();) {
		while (pos < sv.size() &&
		       (sv[pos] == ' ' || sv[pos] == '\t' || sv[pos] == ';'))
			pos++;
		if (pos >= sv.size())
			break;
		size_t end = pos;
		while (end < sv.size() && sv[end] != ';' && sv[end] != ' ' &&
		       sv[end] != '\t')
			end++;
		auto item = sv.substr(pos, end - pos);
		pos = end;
		auto eq = item.find('=');
		if (eq == std::string_view::npos)
			continue;
		if (!fn(item.substr(0, eq), item.substr(eq + 1)))
			return;
	}
}

static bool parse_force_reinit(const gchar *data)
{
	bool result = false;
	for_each_kv(data, [&](std::string_view key,
			      std::string_view val) -> bool {
		if (key != "force_reinit")
			return true;
		result = val == "1" || val == "true" || val == "yes";
		return false;
	});
	return result;
}

static int refresh_attach_session(const gchar *data)
{
	if (__atomic_load_n(&initialized, __ATOMIC_SEQ_CST) != AGENT_READY) {
		SPDLOG_WARN("agent_control: agent not initialized");
		return -EINVAL;
	}
	if (!global_ctx_constructed.load(std::memory_order_acquire)) {
		SPDLOG_WARN("agent_control: global ctx not constructed");
		return -EINVAL;
	}

	std::lock_guard<std::mutex> guard(detach_mutex);
	attach_callback_suppression_scope suppress_callbacks;
	apply_injected_kv_overrides(data);
	(void)ctx_holder.ctx.destroy_all_attach_links();
	ctx_holder.ctx.reset_instantiated_state();

	int failed_handlers = 0;
	int res = ctx_holder.ctx.init_attach_ctx_from_handlers(
		bpftime_get_runtime_config(), &failed_handlers);
	if (res != 0) {
		SPDLOG_ERROR(
			"agent_control: init_attach_ctx_from_handlers failed: {}",
			res);
		return res < 0 ? res : -EINVAL;
	}
	if (!shm_holder.global_shared_memory.add_pid_into_alive_agent_set(
		    getpid())) {
		SPDLOG_ERROR("agent_control: unable to record alive agent pid");
		(void)ctx_holder.ctx.destroy_all_attach_links();
		ctx_holder.ctx.reset_instantiated_state();
		return -EAGAIN;
	}

	if (failed_handlers > 0) {
		SPDLOG_WARN(
			"Attach finished with {} handler(s) not instantiated",
			failed_handlers);
	} else {
		SPDLOG_INFO("Attach successfully");
	}
	start_agent_ipc_server_once();
	return 0;
}

static void apply_injected_kv_overrides(const gchar *data)
{
	for_each_kv(data, [](std::string_view key,
			     std::string_view val) -> bool {
		if (key == "cuda_late_ptx_dir") {
			std::string v(val);
			setenv("BPFTIME_CUDA_LATE_PTX_DIR", v.c_str(), 1);
			SPDLOG_INFO("bpftime-agent: BPFTIME_CUDA_LATE_PTX_DIR={}",
				    v);
		} else if (key == "cuda_disable_cuobjdump") {
			std::string v(val);
			setenv("BPFTIME_CUDA_DISABLE_CUOBJDUMP", v.c_str(), 1);
			SPDLOG_INFO(
				"bpftime-agent: BPFTIME_CUDA_DISABLE_CUOBJDUMP={}",
				v);
		}
		return true;
	});
}

#ifdef BPFTIME_ENABLE_CUDA_ATTACH
namespace bpftime::vm::compat
{
namespace llvm
{
void register_llvm_vm_factory();
} // namespace llvm
} // namespace bpftime::vm::compat
void **(*original___cudaRegisterFatBinary)(void *) = nullptr;

extern "C" void **__cudaRegisterFatBinary(void *fatbin)
{
	try {
		auto orig = try_get_original_func("__cudaRegisterFatBinary",
						 original___cudaRegisterFatBinary);
		if (orig == nullptr)
			return nullptr;
		// We have to register llvmbpf manually, since this function
		// (__cudaRegisterFatBinary) might be called before llvm is registered
		bpftime::vm::compat::llvm::register_llvm_vm_factory();
		return orig(fatbin);
	} catch (...) {
		if (original___cudaRegisterFatBinary != nullptr)
			return original___cudaRegisterFatBinary(fatbin);
	}
	return nullptr;
}
#endif
static void bpftime_agent_main_impl(const gchar *data,
				    gboolean *stay_resident)
{
	if (agent::should_skip_agent_process()) {
		*stay_resident = FALSE;
		return;
	}
	install_agent_bootstrap_logger();
	bool force_reinit = false;
	bool recorded_alive_pid = false;
	bool ctx_constructed = false;
	bool claimed_initialization = false;
	auto init_fail = [&]() noexcept {
		if (!claimed_initialization)
			return;
		try {
			reset_syscall_trace_global();
		} catch (...) {
		}
		if (ctx_constructed) {
			try {
				ctx_holder.destroy();
			} catch (...) {
			}
			ctx_constructed = false;
		}
		global_ctx_constructed.store(false, std::memory_order_release);
		if (recorded_alive_pid) {
			try {
				shm_holder.global_shared_memory
					.remove_pid_from_alive_agent_set(
						getpid());
			} catch (...) {
			}
			recorded_alive_pid = false;
		}
		__atomic_store_n(&initialized, AGENT_UNINITIALIZED,
				 __ATOMIC_SEQ_CST);
	};
	try {
#ifdef __linux__
			// If an agent IPC server is already present in this process,
			// refresh it instead of initializing a new copy.
			if (try_forward_to_existing_agent(data)) {
				*stay_resident = FALSE;
				return;
			}
#endif
			// If another copy of the agent is already present in this
			// process (Frida commonly loads via anonymous fd paths), delegate
			// to it instead of initializing a new copy.
			if (auto existing = find_other_loaded_agent_control();
			    existing) {
				existing(data);
				*stay_resident = FALSE;
				return;
			}

			force_reinit = parse_force_reinit(data);
			{
				int expected = AGENT_UNINITIALIZED;
				if (!__atomic_compare_exchange_n(
					    &initialized, &expected,
					    AGENT_INITIALIZING, false,
					    __ATOMIC_SEQ_CST,
					    __ATOMIC_SEQ_CST)) {
					if (expected != AGENT_READY) {
						*stay_resident = FALSE;
						return;
					}
					if (!force_reinit) {
						SPDLOG_INFO(
							"Agent already initialized, skipping re-initializing..");
						start_agent_ipc_server_once();
						return;
					}
					SPDLOG_INFO(
						"Agent already initialized; force_reinit=1, refreshing attach session..");
					(void)refresh_attach_session(data);
					*stay_resident = TRUE;
					return;
				}
				claimed_initialization = true;
			}

			SPDLOG_DEBUG("Entered bpftime_agent_main");

			// Never block in preload init: fail open on unavailable SHM.
			std::optional<shm_lifecycle_lock> shm_lock(
				std::in_place, get_global_shm_name(), true);
			if (!shm_lock->owns_lock())
				return init_fail();
			try {
				bpftime_initialize_global_shm(
					shm_open_type::SHM_OPEN_ONLY);
			} catch (const std::exception &ex) {
				SPDLOG_ERROR("Unable to initialize shared memory: {}",
					     ex.what());
				return init_fail();
			}
			auto &runtime_config = bpftime_get_runtime_config();
			std::string shm_logger(
				runtime_config.get_logger_output_path());
			const char *env_logger = getenv("BPFTIME_LOG_OUTPUT");
			if (shm_logger != "console" || injected_with_frida ||
			    (env_logger != nullptr &&
			     strcmp(env_logger, "console") == 0)) {
				bpftime_set_logger(shm_logger);
			}
			apply_injected_kv_overrides(data);
			ctx_holder.init();
			ctx_constructed = true;
			global_ctx_constructed.store(
				true, std::memory_order_release);
#if __linux__ && BPFTIME_BUILD_WITH_LIBBPF
			// Register syscall trace impl
			auto syscall_trace_impl =
				std::make_unique<syscall_trace_attach_impl>();
			syscall_trace_impl->set_original_syscall_function(
				orig_hooker);
			auto *syscall_trace_impl_ptr = syscall_trace_impl.get();
			ctx_holder.ctx.register_attach_impl(
				{ ATTACH_SYSCALL_TRACE, ATTACH_SYSCALL_KPROBE,
				  ATTACH_SYSCALL_KRETPROBE },
				std::move(syscall_trace_impl),
				[](const std::string_view &sv, int &err) {
					std::unique_ptr<attach_private_data>
						priv_data =
							std::make_unique<
								syscall_trace_attach_private_data>();
					if (int e =
						    priv_data->initialize_from_string(
							    sv);
					    e < 0) {
						err = e;
						return std::unique_ptr<
							attach_private_data>();
					}
					return priv_data;
				});
#endif
			// Register uprobe attach impl
#if BPFTIME_HAVE_TRAP_UPROBE_ATTACH
			SPDLOG_INFO("Using the trap (breakpoint) uprobe backend");
			ctx_holder.ctx.register_attach_impl(
				{ trap::ATTACH_UPROBE,
				  trap::ATTACH_URETPROBE,
				  trap::ATTACH_UPROBE_OVERRIDE,
				  trap::ATTACH_UREPLACE },
				std::make_unique<trap::trap_attach_impl>(),
				[](const std::string_view &sv, int &err)
					-> std::unique_ptr<attach_private_data> {
					auto priv_data = std::make_unique<
						trap::trap_attach_private_data>();
					if (int e =
						    priv_data->initialize_from_string(
							    sv);
					    e < 0) {
						err = e;
						return std::unique_ptr<
							attach_private_data>();
					}
					return priv_data;
				});
#elif BPFTIME_HAVE_FRIDA_ATTACH
			ctx_holder.ctx.register_attach_impl(
				{ ATTACH_UPROBE, ATTACH_URETPROBE,
				  ATTACH_UPROBE_OVERRIDE, ATTACH_UREPLACE },
				std::make_unique<attach::frida_attach_impl>(
					runtime_config.enable_frida_fuzzy_backtracer),
				[](const std::string_view &sv, int &err)
					-> std::unique_ptr<attach_private_data> {
					auto priv_data = std::make_unique<
						frida_attach_private_data>();
					if (int e =
						    priv_data->initialize_from_string(
							    sv);
					    e < 0) {
						err = e;
						return std::unique_ptr<
							attach_private_data>();
					}
					return priv_data;
				});
#else
			SPDLOG_WARN("No uprobe attach backend available");
#endif

#ifdef BPFTIME_ENABLE_CUDA_ATTACH
			// Register cuda attach impl
			ctx_holder.ctx.register_attach_impl(
				{ ATTACH_CUDA_PROBE, ATTACH_CUDA_RETPROBE },
				std::make_unique<attach::nv_attach_impl>(),
				[](const std::string_view &sv, int &err) {
					std::unique_ptr<attach_private_data>
						priv_data =
							std::make_unique<
								nv_attach_private_data>();
					if (int e =
						    priv_data->initialize_from_string(
							    sv);
					    e < 0) {
						err = e;
						return std::unique_ptr<
							attach_private_data>();
					}
					return priv_data;
				});
#endif
#if __linux__ && BPFTIME_BUILD_WITH_LIBBPF
			syscall_trace_impl_ptr->set_to_global();
#endif
			SPDLOG_INFO("Initializing agent..");

			int failed_handlers = 0;
			try {
				attach_callback_suppression_scope suppress_callbacks;
				int res = ctx_holder.ctx
						  .init_attach_ctx_from_handlers(
							  runtime_config,
							  &failed_handlers);
				if (res != 0) {
					SPDLOG_INFO(
						"Failed to initialize attach context, exiting..");
					init_fail();
					return;
				}
			} catch (const std::exception &ex) {
				SPDLOG_ERROR(
					"Unable to instantiate handlers: {}",
					ex.what());
				init_fail();
				return;
			}
#if __linux__ && BPFTIME_BUILD_WITH_LIBBPF
			syscall_trace_impl_ptr->dispatch_process_exec();
#endif
			srand(std::random_device()());

			if (!shm_holder.global_shared_memory
				     .add_pid_into_alive_agent_set(getpid())) {
				SPDLOG_ERROR("Unable to record alive agent pid");
				init_fail();
				return;
			}
			recorded_alive_pid = true;
			shm_lock.reset();
			start_agent_ipc_server_once();
			std::atexit(stop_syscall_dispatch_at_exit);
			/* We don't want our library to be unloaded after we return. */
			*stay_resident = TRUE;
			setenv("BPFTIME_USED", "1", 0);
			__atomic_store_n(&initialized, AGENT_READY,
					 __ATOMIC_SEQ_CST);
			try {
				SPDLOG_DEBUG("Set environment variable BPFTIME_USED");
			} catch (...) {
			}
			try {
				if (failed_handlers > 0) {
					SPDLOG_WARN(
						"Attach finished with {} handler(s) not instantiated",
						failed_handlers);
				} else {
					SPDLOG_INFO("Attach successfully");
				}
			} catch (...) {
			}
		} catch (const std::exception &) {
			init_fail();
		} catch (...) {
			init_fail();
		}
}

extern "C" void bpftime_agent_main(const gchar *data, gboolean *stay_resident)
{
	injected_with_frida = true;
	bpftime_agent_main_impl(data, stay_resident);
}

// using definition for libbpf for syscall issues
// maybe should separate libbpf and kernel features separately
#if __linux__ && BPFTIME_BUILD_WITH_LIBBPF
extern "C" int64_t syscall_callback(int64_t sys_nr, int64_t arg1, int64_t arg2,
				    int64_t arg3, int64_t arg4, int64_t arg5,
				    int64_t arg6, int64_t user_ip,
				    int64_t user_sp, int64_t user_bp)
{
	try {
		auto *impl =
			bpftime::attach::global_syscall_trace_attach_impl.load(
				std::memory_order_acquire);
		if (impl != nullptr) {
			return impl->dispatch_syscall(
				sys_nr, arg1, arg2, arg3, arg4, arg5, arg6,
				user_ip, user_sp, user_bp);
		}
	} catch (...) {
	}
	if (orig_hooker != nullptr)
		return orig_hooker(sys_nr, arg1, arg2, arg3, arg4, arg5, arg6,
				   user_ip, user_sp, user_bp);
	return -ENOSYS;
}

extern "C" void
_bpftime__setup_syscall_trace_callback(syscall_hooker_func_t *hooker)
{
	if (hooker == nullptr)
		return;
	// Text-transformer entrypoint: a rejected process keeps its original
	// hook and must not log either, since the bootstrap logger never ran.
	if (agent::should_skip_agent_process())
		return;
	orig_hooker = *hooker;
	*hooker = &syscall_callback;
	gboolean val = FALSE;
	try {
		bpftime_agent_main_impl("", &val);
		SPDLOG_INFO("Agent syscall trace setup exiting..");
	} catch (...) {
	}
}
#endif
