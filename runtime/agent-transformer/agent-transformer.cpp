#include "spdlog/cfg/env.h"
#include "spdlog/spdlog.h"
#include <cassert>
#include <cstdlib>
#include <dlfcn.h>
#include <frida-gum.h>
#include "text_segment_transformer.hpp"
#include <spdlog/cfg/env.h>
#include <string>
#include <frida-gum.h>

using main_func_t = int (*)(int, char **, char **);
using shm_destroy_func_t = void (*)(void);

static main_func_t orig_main_func = nullptr;
static shm_destroy_func_t shm_destroy_func = nullptr;
extern "C" void bpftime_agent_main(const gchar *data, gboolean *stay_resident);

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
	spdlog::info("Entering bpftime agent");
	orig_main_func = main;
	using this_func_t = decltype(&__libc_start_main);
	this_func_t orig = (this_func_t)dlsym(RTLD_NEXT, "__libc_start_main");

	return orig(bpftime_hooked_main, argc, argv, init, fini, rtld_fini,
		    stack_end);
}

extern "C" void bpftime_agent_main(const gchar *data, gboolean *stay_resident)
{
	spdlog::cfg::load_env_levels();
	/* We don't want to our library to be unloaded after we return. */
	*stay_resident = TRUE;

	const char *agent_so = getenv("AGENT_SO");
	if (agent_so == nullptr) {
		if (std::string(data) != "") {
			spdlog::info("Using agent path from frida data..");
			agent_so = data;
		}
	}
	assert(agent_so &&
	       "Please set AGENT_SO to the bpftime-agent when use this tranformer");
	spdlog::info("Using agent {}", agent_so);
	cs_arch_register_x86();
	bpftime::setup_syscall_tracer();
	spdlog::debug("Loading dynamic library..");
	auto next_handle =
		dlmopen(LM_ID_NEWLM, agent_so, RTLD_NOW | RTLD_LOCAL);
	if (next_handle == nullptr) {
		spdlog::error("Failed to open agent: {}", dlerror());
		exit(1);
	}
	shm_destroy_func = (shm_destroy_func_t)dlsym(
		next_handle, "bpftime_destroy_global_shm");
	auto entry_func = (void (*)(syscall_hooker_func_t *))dlsym(
		next_handle, "__c_abi_setup_syscall_trace_callback");

	assert(entry_func &&
	       "Malformed agent so, expected symbol __c_abi_setup_syscall_trace_callback");
	syscall_hooker_func_t orig_syscall_hooker_func =
		bpftime::get_call_hook();
	entry_func(&orig_syscall_hooker_func);
	bpftime::set_call_hook(orig_syscall_hooker_func);
	spdlog::info("Transformer exiting, trace will be usable now");
}
