#include "attach_private_data.hpp"
#include "bpf_attach_ctx.hpp"
#include "bpftime_shm_internal.hpp"
#include "frida_attach_private_data.hpp"
#include "frida_uprobe_attach_impl.hpp"

#include "spdlog/common.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/stdout_sinks.h"
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
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <frida-gum.h>
#include <cstdint>
#include <dlfcn.h>
#include <link.h>
#ifdef __linux__
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/types.h>
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

static int initialized = 0;

using agent_control_fn_t = void (*)(const gchar *);

bool injected_with_frida = true;

static std::atomic<uint64_t> auto_refresh_epoch{ 0 };
static std::mutex auto_refresh_thread_mutex;
static std::thread auto_refresh_thread;
static std::atomic<pid_t> auto_refresh_loader_pid{ -1 };

static std::once_flag agent_ipc_once;
static std::atomic<bool> agent_ipc_stop{ false };
#ifdef __linux__
static int agent_ipc_fd = -1;
static std::thread agent_ipc_thread;
#endif

static int detach_pipe_fds[2] = { -1, -1 };
static std::atomic<bool> detach_thread_started{ false };
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
static bool refresh_attach_session(const gchar *data);
static void perform_detach();

extern "C" __attribute__((visibility("default"))) void
bpftime_agent_control(const gchar *data)
{
	refresh_attach_session(data);
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
static bool make_agent_ipc_addr(pid_t pid, sockaddr_un &addr, socklen_t &len)
{
	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	std::string name = "bpftime-agent-" + std::to_string((int)pid);
	if (name.size() + 1 > sizeof(addr.sun_path))
		return false;
	addr.sun_path[0] = '\0';
	memcpy(addr.sun_path + 1, name.data(), name.size());
	len = (socklen_t)(offsetof(sockaddr_un, sun_path) + 1 + name.size());
	return true;
}

static bool try_send_agent_ipc(pid_t pid, const std::string &request,
			       std::string *response_out = nullptr)
{
	int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
	if (fd < 0)
		return false;
	sockaddr_un addr {};
	socklen_t len = 0;
	if (!make_agent_ipc_addr(pid, addr, len)) {
		::close(fd);
		return false;
	}
	if (::connect(fd, (sockaddr *)&addr, len) != 0) {
		::close(fd);
		return false;
	}
	const char *buf = request.data();
	size_t left = request.size();
	while (left > 0) {
		ssize_t n = ::send(fd, buf, left, MSG_NOSIGNAL);
		if (n < 0) {
			::close(fd);
			return false;
		}
		buf += (size_t)n;
		left -= (size_t)n;
	}
	::shutdown(fd, SHUT_WR);
	if (response_out) {
		std::string resp;
		char rbuf[4096];
		for (;;) {
			ssize_t n = ::recv(fd, rbuf, sizeof(rbuf), 0);
			if (n <= 0)
				break;
			resp.append(rbuf, rbuf + n);
		}
		*response_out = std::move(resp);
	}
	::close(fd);
	return true;
}

static bool try_forward_to_existing_agent(const gchar *data)
{
	std::string req("refresh ");
	if (data != nullptr)
		req += data;
	return try_send_agent_ipc(getpid(), req);
}
#endif

static void start_agent_ipc_server_once()
{
#ifdef __linux__
	std::call_once(agent_ipc_once, []() {
		sockaddr_un addr {};
		socklen_t len = 0;
		if (!make_agent_ipc_addr(getpid(), addr, len)) {
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
				struct ucred cred {};
				socklen_t clen = sizeof(cred);
				if (getsockopt(cfd, SOL_SOCKET, SO_PEERCRED, &cred,
					       &clen) == 0) {
					if (cred.uid != 0 &&
					    (uid_t)cred.uid != geteuid()) {
						::close(cfd);
						continue;
					}
				}
				std::string req;
				char buf[4096];
				for (;;) {
					ssize_t n = ::recv(cfd, buf, sizeof(buf), 0);
					if (n <= 0)
						break;
					req.append(buf, buf + n);
					if (req.size() > (1 << 20))
						break;
				}
				auto trim = [](std::string &s) {
					while (!s.empty() &&
					       (s.back() == '\n' || s.back() == '\r' ||
						s.back() == '\0'))
						s.pop_back();
				};
				trim(req);
				if (req.rfind("refresh ", 0) == 0) {
					std::string arg = req.substr(strlen("refresh "));
					refresh_attach_session(arg.c_str());
					::send(cfd, "ok\n", 3, MSG_NOSIGNAL);
				} else if (req == "detach") {
					perform_detach();
					::send(cfd, "ok\n", 3, MSG_NOSIGNAL);
				} else if (req == "status") {
					uint64_t epoch =
						shm_holder.global_shared_memory
							.read_stable_epoch_seq();
					std::string out =
						"epoch_seq=" + std::to_string(epoch) + "\n";
					::send(cfd, out.data(), out.size(), MSG_NOSIGNAL);
				} else {
					::send(cfd, "unknown\n", 8, MSG_NOSIGNAL);
				}
				::close(cfd);
			}
		});
		std::atexit([]() {
			agent_ipc_stop.store(true, std::memory_order_release);
			if (agent_ipc_fd >= 0) {
				::close(agent_ipc_fd);
				agent_ipc_fd = -1;
			}
			if (agent_ipc_thread.joinable())
				agent_ipc_thread.detach();
		});
		SPDLOG_INFO("agent ipc: listening (abstract) for pid {}", (int)getpid());
	});
#else
#endif
}

static void perform_detach()
{
	std::lock_guard<std::mutex> detach_guard(detach_mutex);
	if (!global_ctx_constructed.load(std::memory_order_acquire)) {
		__atomic_store_n(&initialized, 0, __ATOMIC_SEQ_CST);
		return;
	}
	SPDLOG_INFO("Detaching..");

	auto_refresh_epoch.fetch_add(1, std::memory_order_acq_rel);
	{
		std::lock_guard<std::mutex> guard(auto_refresh_thread_mutex);
		if (auto_refresh_thread.joinable())
			auto_refresh_thread.join();
	}

	int detach_err = ctx_holder.ctx.destroy_all_attach_links();
	if (detach_err < 0) {
		SPDLOG_ERROR("Unable to detach cleanly: {}", detach_err);
	}
	ctx_holder.ctx.reset_instantiated_state();
	SPDLOG_DEBUG("Detaching done");
	bpftime_logger_flush();
}

static void stop_auto_refresh_at_exit()
{
	auto_refresh_epoch.fetch_add(1, std::memory_order_acq_rel);
	std::lock_guard<std::mutex> guard(auto_refresh_thread_mutex);
	if (auto_refresh_thread.joinable())
		auto_refresh_thread.join();
}

static void ensure_detach_worker_started()
{
	bool expected = false;
	if (!detach_thread_started.compare_exchange_strong(
		    expected, true, std::memory_order_acq_rel,
		    std::memory_order_acquire)) {
		return;
	}
	int fds[2];
#ifdef __linux__
	if (pipe2(fds, O_CLOEXEC) != 0) {
		SPDLOG_WARN("pipe2 failed, detach by SIGUSR1 will be disabled");
		detach_thread_started.store(false, std::memory_order_release);
		return;
	}
#else
	if (pipe(fds) != 0) {
		SPDLOG_WARN("pipe failed, detach by SIGUSR1 will be disabled");
		detach_thread_started.store(false, std::memory_order_release);
		return;
	}
#endif
	detach_pipe_fds[0] = fds[0];
	detach_pipe_fds[1] = fds[1];
	std::thread([]() {
		for (;;) {
			uint8_t buf[16];
			ssize_t n = read(detach_pipe_fds[0], buf, sizeof(buf));
			if (n <= 0) {
				return;
			}
			if (__atomic_load_n(&initialized, __ATOMIC_SEQ_CST) != 1) {
				continue;
			}
			perform_detach();
		}
	}).detach();
}

syscall_hooker_func_t orig_hooker;

extern "C" void bpftime_agent_main(const gchar *data, gboolean *stay_resident);

extern "C" int bpftime_hooked_main(int argc, char **argv, char **envp)
{
	int stay_resident = 0;
	injected_with_frida = false;
	bpftime_agent_main("", &stay_resident);
	int ret = orig_main_func(argc, argv, envp);
	return ret;
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

	return orig(bpftime_hooked_main, argc, argv, init, fini, rtld_fini,
		    stack_end);
}
static void sig_handler_sigusr1_detach(int)
{
	auto_refresh_epoch.fetch_add(1, std::memory_order_acq_rel);
	int fd = detach_pipe_fds[1];
	if (fd >= 0) {
		uint8_t one = 1;
		if (write(fd, &one, 1) < 0) {
		}
	}
}

static int parse_auto_refresh_ms(const gchar *data)
{
	if (data == nullptr || data[0] == '\0')
		return 0;
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
		auto key = item.substr(0, eq);
		auto val = item.substr(eq + 1);
		if (key != "auto_refresh_ms")
			continue;
		int out = 0;
		try {
			out = std::stoi(std::string(val));
		} catch (...) {
			return 0;
		}
		if (out < 0)
			return 0;
		return out;
	}
	return 0;
}

static pid_t parse_loader_pid(const gchar *data)
{
	if (data == nullptr || data[0] == '\0')
		return -1;
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
		auto key = item.substr(0, eq);
		auto val = item.substr(eq + 1);
		if (key != "loader_pid")
			continue;
		try {
			long long out = std::stoll(std::string(val));
			if (out <= 0)
				return -1;
			if (out > std::numeric_limits<pid_t>::max())
				return -1;
			return (pid_t)out;
		} catch (...) {
			return -1;
		}
	}
	return -1;
}

static bool parse_force_reinit(const gchar *data)
{
	if (data == nullptr || data[0] == '\0')
		return false;
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
		auto key = item.substr(0, eq);
		auto val = item.substr(eq + 1);
		if (key != "force_reinit")
			continue;
		return val == "1" || val == "true" || val == "yes";
	}
	return false;
}

static bool refresh_attach_session(const gchar *data)
{
	if (__atomic_load_n(&initialized, __ATOMIC_SEQ_CST) != 1) {
		SPDLOG_WARN("agent_control: agent not initialized");
		return false;
	}
	if (!global_ctx_constructed.load(std::memory_order_acquire)) {
		SPDLOG_WARN("agent_control: global ctx not constructed");
		return false;
	}

	std::lock_guard<std::mutex> guard(detach_mutex);
	apply_injected_kv_overrides(data);
	auto_refresh_epoch.fetch_add(1, std::memory_order_acq_rel);
		{
			std::lock_guard<std::mutex> tguard(auto_refresh_thread_mutex);
			if (auto_refresh_thread.joinable())
				auto_refresh_thread.join();
		}
		ctx_holder.ctx.destroy_all_attach_links();
		ctx_holder.ctx.reset_instantiated_state();

	int res =
		ctx_holder.ctx.init_attach_ctx_from_handlers(bpftime_get_agent_config());
	if (res != 0) {
		SPDLOG_ERROR(
			"agent_control: init_attach_ctx_from_handlers failed: {}",
			res);
		return false;
	}

	int auto_refresh_ms = parse_auto_refresh_ms(data);
	pid_t loader_pid = parse_loader_pid(data);
	if (auto_refresh_ms > 0) {
		auto_refresh_loader_pid.store(loader_pid,
					      std::memory_order_release);
		SPDLOG_INFO("Starting auto-refresh thread ({} ms, loader_pid={})",
			    auto_refresh_ms, (int)loader_pid);
		uint64_t epoch =
			auto_refresh_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
		std::lock_guard<std::mutex> tguard(auto_refresh_thread_mutex);
		if (auto_refresh_thread.joinable())
			auto_refresh_thread.join();
		auto_refresh_thread = std::thread([auto_refresh_ms, epoch]() {
			for (;;) {
				if (auto_refresh_epoch.load(
					    std::memory_order_acquire) != epoch) {
					return;
				}
#if __linux__
				pid_t lp = auto_refresh_loader_pid.load(
					std::memory_order_acquire);
				if (lp > 0) {
					if (::kill(lp, 0) != 0 &&
					    errno == ESRCH) {
						SPDLOG_INFO(
							"Auto-refresh: loader pid {} is gone; stopping",
							(int)lp);
						return;
					}
				}
#endif
				std::this_thread::sleep_for(
					std::chrono::milliseconds(auto_refresh_ms));
				try {
					ctx_holder.ctx.init_attach_ctx_from_handlers(
						bpftime_get_agent_config());
				} catch (const std::exception &ex) {
					SPDLOG_DEBUG(
						"Auto-refresh: init_attach_ctx_from_handlers failed: {}",
						ex.what());
				}
			}
		});
		std::atexit(stop_auto_refresh_at_exit);
	}

	SPDLOG_INFO("Attach successfully");
	start_agent_ipc_server_once();
	return true;
}

static void apply_injected_kv_overrides(const gchar *data)
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
		auto key = item.substr(0, eq);
		auto val = item.substr(eq + 1);
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
	}
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
		bpftime::vm::compat::llvm::register_llvm_vm_factory();
		return orig(fatbin);
	} catch (const std::exception &ex) {
		fprintf(stderr,
			"bpftime-agent: __cudaRegisterFatBinary wrapper failed: %s\n",
			ex.what());
	} catch (...) {
		fprintf(stderr,
			"bpftime-agent: __cudaRegisterFatBinary wrapper failed: unknown error\n");
	}
	return nullptr;
}
#endif
extern "C" void bpftime_agent_main(const gchar *data, gboolean *stay_resident)
{
	try {
#ifdef __linux__
			if (try_forward_to_existing_agent(data)) {
				*stay_resident = FALSE;
				return;
			}
#endif
			if (auto existing = find_other_loaded_agent_control();
			    existing) {
				existing(data);
				*stay_resident = FALSE;
				return;
			}

			bool force_reinit = parse_force_reinit(data);
			bool recorded_alive_pid = false;
			bool ctx_constructed = false;
			auto init_fail = [&]() {
				if (ctx_constructed) {
					ctx_holder.destroy();
					ctx_constructed = false;
				}
				global_ctx_constructed.store(false,
							     std::memory_order_release);
				if (recorded_alive_pid) {
					shm_holder.global_shared_memory
						.remove_pid_from_alive_agent_set(
							getpid());
					recorded_alive_pid = false;
				}
				__atomic_store_n(&initialized, 0,
						 __ATOMIC_SEQ_CST);
			};
			{
				int expected = 0;
				if (!__atomic_compare_exchange_n(
					    &initialized, &expected, 1, false,
					    __ATOMIC_SEQ_CST,
					    __ATOMIC_SEQ_CST)) {
					if (!force_reinit) {
						SPDLOG_INFO(
							"Agent already initialized, skipping re-initializing..");
						start_agent_ipc_server_once();
						return;
					}
						SPDLOG_INFO(
							"Agent already initialized; force_reinit=1, refreshing attach session..");
						refresh_attach_session(data);
						*stay_resident = TRUE;
						return;
					}
			}

				SPDLOG_DEBUG("Entered bpftime_agent_main");
				SPDLOG_DEBUG("Registering signal handler");

				srand(std::random_device()());
				ensure_detach_worker_started();
				signal(SIGUSR1, sig_handler_sigusr1_detach);

				std::string last_err;
				bool shm_ok = false;
			for (int attempt = 0; attempt < 60; attempt++) {
				try {
					bpftime_initialize_global_shm(
						shm_open_type::SHM_OPEN_ONLY);
					shm_ok = true;
					break;
				} catch (const std::exception &ex) {
					last_err = ex.what();
					std::this_thread::sleep_for(
						std::chrono::milliseconds(
							50));
				}
			}
			if (!shm_ok) {
				SPDLOG_ERROR(
					"Unable to initialize shared memory: {}",
					last_err);
				init_fail();
				return;
			}
			auto &runtime_config = bpftime_get_agent_config();
			bpftime_set_logger(std::string(
				runtime_config.get_logger_output_path()));
			apply_injected_kv_overrides(data);
			if (injected_with_frida) {
				shm_holder.global_shared_memory
					.add_pid_into_alive_agent_set(getpid());
				recorded_alive_pid = true;
			}
			ctx_holder.init();
			ctx_constructed = true;
			global_ctx_constructed.store(
				true, std::memory_order_release);
#if __linux__ && BPFTIME_BUILD_WITH_LIBBPF
			auto syscall_trace_impl =
				std::make_unique<syscall_trace_attach_impl>();
			syscall_trace_impl->set_original_syscall_function(
				orig_hooker);
			syscall_trace_impl->set_to_global();
			ctx_holder.ctx.register_attach_impl(
				{ ATTACH_SYSCALL_TRACE },
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
			ctx_holder.ctx.register_attach_impl(
				{ ATTACH_UPROBE, ATTACH_URETPROBE,
				  ATTACH_UPROBE_OVERRIDE, ATTACH_UREPLACE },
				std::make_unique<attach::frida_attach_impl>(),
				[](const std::string_view &sv, int &err) {
					std::unique_ptr<attach_private_data>
						priv_data =
							std::make_unique<
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

#ifdef BPFTIME_ENABLE_CUDA_ATTACH
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
			SPDLOG_INFO("Initializing agent..");
			*stay_resident = TRUE;

			setenv("BPFTIME_USED", "1", 0);
			SPDLOG_DEBUG("Set environment variable BPFTIME_USED");
			try {
				int res = ctx_holder.ctx
						  .init_attach_ctx_from_handlers(
							  runtime_config);
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

			start_agent_ipc_server_once();

			int auto_refresh_ms = parse_auto_refresh_ms(data);
			pid_t loader_pid = parse_loader_pid(data);
			if (auto_refresh_ms > 0) {
				auto_refresh_loader_pid.store(
					loader_pid,
					std::memory_order_release);
				SPDLOG_INFO(
					"Starting auto-refresh thread ({} ms, loader_pid={})",
					auto_refresh_ms, (int)loader_pid);
				uint64_t epoch =
					auto_refresh_epoch.fetch_add(
						1, std::memory_order_acq_rel) +
					1;
				std::lock_guard<std::mutex> guard(
					auto_refresh_thread_mutex);
				if (auto_refresh_thread.joinable())
					auto_refresh_thread.join();
				auto_refresh_thread = std::thread(
					[auto_refresh_ms, epoch]() {
						for (;;) {
							if (auto_refresh_epoch.load(
								    std::memory_order_acquire) !=
							    epoch) {
								return;
							}
#if __linux__
							pid_t lp =
								auto_refresh_loader_pid.load(
									std::memory_order_acquire);
							if (lp > 0) {
								if (::kill(lp, 0) != 0 &&
								    errno == ESRCH) {
									SPDLOG_INFO(
										"Auto-refresh: loader pid {} is gone; stopping",
										(int)lp);
									return;
								}
							}
#endif
							std::this_thread::sleep_for(
								std::chrono::milliseconds(
									auto_refresh_ms));
							try {
								ctx_holder.ctx
									.init_attach_ctx_from_handlers(
										bpftime_get_agent_config());
							} catch (const std::exception &ex) {
								SPDLOG_DEBUG(
									"Auto-refresh: init_attach_ctx_from_handlers failed: {}",
									ex.what());
							}
						}
					});
				std::atexit(stop_auto_refresh_at_exit);
			}

			SPDLOG_INFO("Attach successfully");
		} catch (const std::exception &ex) {
			fprintf(stderr,
				"bpftime-agent: bpftime_agent_main failed: %s\n",
				ex.what());
			__atomic_store_n(&initialized, 0, __ATOMIC_SEQ_CST);
	} catch (...) {
			fprintf(stderr,
				"bpftime-agent: bpftime_agent_main failed: unknown error\n");
			__atomic_store_n(&initialized, 0, __ATOMIC_SEQ_CST);
		}
}

#if __linux__ && BPFTIME_BUILD_WITH_LIBBPF
extern "C" int64_t syscall_callback(int64_t sys_nr, int64_t arg1, int64_t arg2,
				    int64_t arg3, int64_t arg4, int64_t arg5,
				    int64_t arg6)
{
	return bpftime::attach::global_syscall_trace_attach_impl.value()
		->dispatch_syscall(sys_nr, arg1, arg2, arg3, arg4, arg5, arg6);
}

extern "C" void
_bpftime__setup_syscall_trace_callback(syscall_hooker_func_t *hooker)
{
	orig_hooker = *hooker;
	*hooker = &syscall_callback;
	gboolean val;
	bpftime_agent_main("", &val);
	SPDLOG_INFO("Agent syscall trace setup exiting..");
}
#endif
