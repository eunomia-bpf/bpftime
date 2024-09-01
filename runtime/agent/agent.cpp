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
#include <exception>
#include <fcntl.h>
#include <memory>
#include <pthread.h>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <frida-gum.h>
#include <cstdint>
#include <dlfcn.h>
#include "bpftime_shm.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>
#if __linux__ && BPFTIME_BUILD_WITH_LIBBPF
#include "syscall_trace_attach_impl.hpp"
#include "syscall_trace_attach_private_data.hpp"
#endif

using namespace bpftime;
using namespace bpftime::attach;
using main_func_t = int (*)(int, char **, char **);

static main_func_t orig_main_func = nullptr;

// Whether this injected process was operated through frida?
// Defaults to true. If __libc_start_main was called, it should be set to false;
// Besides, if agent was loaded by text-transformer, this variable will be set
// by text-transformer
bool injected_with_frida = true;

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

syscall_hooker_func_t orig_hooker;

extern "C" void bpftime_agent_main(const gchar *data, gboolean *stay_resident);

extern "C" int bpftime_hooked_main(int argc, char **argv, char **envp)
{
	int stay_resident = 0;
	injected_with_frida = false;
	bpftime_agent_main("", &stay_resident);
	int ret = orig_main_func(argc, argv, envp);
	ctx_holder.destroy();
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

static void sig_handler_sigusr1(int sig)
{
	SPDLOG_INFO("Detaching..");
	if (int err = ctx_holder.ctx.destroy_all_attach_links(); err < 0) {
		SPDLOG_ERROR("Unable to detach: {}", err);
		return;
	}
	shm_holder.global_shared_memory.remove_pid_from_alive_agent_set(
		getpid());
	SPDLOG_DEBUG("Detaching done");
	bpftime_logger_flush();
}

extern "C" void bpftime_agent_main(const gchar *data, gboolean *stay_resident)
{
	auto runtime_config = bpftime_get_agent_config();
	bpftime_set_logger(runtime_config.logger_output_path);
	SPDLOG_DEBUG("Entered bpftime_agent_main");
	SPDLOG_DEBUG("Registering signal handler");
	// We use SIGUSR1 to indicate the detaching
	signal(SIGUSR1, sig_handler_sigusr1);
	try {
		// If we are unable to initialize shared memory..
		bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);
	} catch (std::exception &ex) {
		SPDLOG_ERROR("Unable to initialize shared memory: {}",
			     ex.what());
		return;
	}
	// Only agents injected through frida could be detached
	if (injected_with_frida) {
		// Record the pid
		shm_holder.global_shared_memory.add_pid_into_alive_agent_set(
			getpid());
	}
	ctx_holder.init();
	#if __linux__ && BPFTIME_BUILD_WITH_LIBBPF
	// Register syscall trace impl
	auto syscall_trace_impl = std::make_unique<syscall_trace_attach_impl>();
	syscall_trace_impl->set_original_syscall_function(orig_hooker);
	syscall_trace_impl->set_to_global();
	ctx_holder.ctx.register_attach_impl(
		{ ATTACH_SYSCALL_TRACE }, std::move(syscall_trace_impl),
		[](const std::string_view &sv, int &err) {
			std::unique_ptr<attach_private_data> priv_data =
				std::make_unique<
					syscall_trace_attach_private_data>();
			if (int e = priv_data->initialize_from_string(sv);
			    e < 0) {
				err = e;
				return std::unique_ptr<attach_private_data>();
			}
			return priv_data;
		});
	#endif
	// Register uprobe attach impl
	ctx_holder.ctx.register_attach_impl(
		{ ATTACH_UPROBE, ATTACH_URETPROBE, ATTACH_UPROBE_OVERRIDE,
		  ATTACH_UREPLACE },
		std::make_unique<attach::frida_attach_impl>(),
		[](const std::string_view &sv, int &err) {
			std::unique_ptr<attach_private_data> priv_data =
				std::make_unique<frida_attach_private_data>();
			if (int e = priv_data->initialize_from_string(sv);
			    e < 0) {
				err = e;
				return std::unique_ptr<attach_private_data>();
			}
			return priv_data;
		});
	SPDLOG_INFO("Initializing agent..");
	/* We don't want to our library to be unloaded after we return. */
	*stay_resident = TRUE;

	int res = 1;
	setenv("BPFTIME_USED", "1", 0);
	SPDLOG_DEBUG("Set environment variable BPFTIME_USED");
	try {
		res = ctx_holder.ctx.init_attach_ctx_from_handlers(
			runtime_config);
		if (res != 0) {
			SPDLOG_INFO("Failed to initialize attach context, exiting..");
			return;
		}
	} catch (std::exception &ex) {
		SPDLOG_ERROR("Unable to instantiate handlers with error: {}", ex.what());
		return;
	}
	SPDLOG_INFO("Attach successfully");
}

// using definition for libbpf for syscall issues
// maybe should separate libbpf and kernel features separately
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