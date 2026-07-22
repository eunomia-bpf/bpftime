#include "bpftime_logger.hpp"
#include "spdlog/spdlog.h"
#include <cstdlib>
#include <dlfcn.h>
#include <frida-gum.h>
#include "text_segment_transformer.hpp"
#include <string>

using main_func_t = int (*)(int, char **, char **);
using shm_destroy_func_t = void (*)(void);

static main_func_t orig_main_func = nullptr;
static shm_destroy_func_t shm_destroy_func = nullptr;

// Whether syscall server was injected using frida. Defaults to true. If
// __libc_start_main was called, it will be set to false
static bool injected_with_frida = true;
extern "C" void bpftime_agent_main(const gchar *data, gboolean *stay_resident);

static void configure_transformer_logger() noexcept
{
	const char *target = std::getenv("BPFTIME_LOG_OUTPUT");
	bpftime::bpftime_set_logger(
		target != nullptr ? target : "~/.bpftime/runtime.log");
}

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
	configure_transformer_logger();
	orig_main_func = main;
	using this_func_t = decltype(&__libc_start_main);
	this_func_t orig = (this_func_t)dlsym(RTLD_NEXT, "__libc_start_main");

	return orig(bpftime_hooked_main, argc, argv, init, fini, rtld_fini,
		    stack_end);
}

static void bpftime_agent_main_impl(const gchar *data, gboolean *stay_resident)
{
	configure_transformer_logger();
	/* We don't want to our library to be unloaded after we return. */
	if (stay_resident != nullptr)
		*stay_resident = TRUE;

	const char *agent_so = getenv("AGENT_SO");
	if (agent_so == nullptr) {
		if (data != nullptr && data[0] != '\0') {
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
	if (!bpftime::setup_syscall_tracer())
		return;
	SPDLOG_DEBUG("Loading dynamic library..");
	auto next_handle =
		dlmopen(LM_ID_NEWLM, agent_so, RTLD_NOW | RTLD_LOCAL);
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

	if (!entry_func) {
		SPDLOG_CRITICAL(
			"Malformed agent so, expected symbol _bpftime__setup_syscall_hooker_callback");
		return;
	}
	syscall_hooker_func_t orig_syscall_hooker_func =
		bpftime::get_call_hook();
	entry_func(&orig_syscall_hooker_func);
	bpftime::set_call_hook(orig_syscall_hooker_func);
	SPDLOG_DEBUG("Transformer exiting, syscall trace is usable now");
}

extern "C" void bpftime_agent_main(const gchar *data, gboolean *stay_resident)
{
	try {
		bpftime_agent_main_impl(data, stay_resident);
	} catch (const std::exception &error) {
		try {
			SPDLOG_ERROR("Failed to initialize transformer: {}",
				     error.what());
		} catch (...) {
		}
	} catch (...) {
	}
}
