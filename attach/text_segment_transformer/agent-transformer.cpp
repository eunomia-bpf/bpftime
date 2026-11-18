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
#include <mutex>
#include <optional>
#include "text_segment_transformer.hpp"
#if defined(__linux__) && defined(__x86_64__)
#include "seccomp_tracer.hpp"
#endif
#include <string>
#include <thread>

using main_func_t = int (*)(int, char **, char **);

static main_func_t orig_main_func = nullptr;

// Whether syscall server was injected using frida. Defaults to true. If
// __libc_start_main was called, it will be set to false
static bool injected_with_frida = true;
extern "C" void bpftime_agent_main(const gchar *data, gboolean *stay_resident);
extern "C" int64_t call_orig_syscall(int64_t sys_nr, int64_t arg1,
				      int64_t arg2, int64_t arg3, int64_t arg4,
				      int64_t arg5, int64_t arg6,
				      int64_t user_ip, int64_t user_sp,
				      int64_t user_bp);

extern "C" int bpftime_hooked_main(int argc, char **argv, char **envp)
{
	int stay_resident = 0;
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
	// Set the `injected_with_frida` flag when the agent exports it. The
	// syscall-server does not, so a missing symbol only skips the assignment;
	// returning here would also skip seccomp setup and break rootless tracing.
	bool *injected_with_frida__agent =
		(bool *)dlsym(next_handle, "injected_with_frida");
	if (injected_with_frida__agent)
		*injected_with_frida__agent = injected_with_frida;
	else
		SPDLOG_WARN("Agent does not expose injected_with_frida");
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
		if (!bpftime::seccomp::install_seccomp_tracer()) {
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
