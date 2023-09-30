#include "spdlog/cfg/env.h"
#include "spdlog/spdlog.h"
#include <cassert>
#include <cstdlib>
#include <dlfcn.h>
#include <cstdio>
#include <frida-gum.h>
#include "text_segment_transformer.hpp"
#include <iostream>
#include <ostream>
#include <spdlog/cfg/env.h>
#include <string>
using putchar_func = int (*)(int c);
using puts_func_t = int (*)(const char *);

static puts_func_t orig_puts_func = nullptr;

// using putchar_func as a flag to indicate whether the agent has been init
static putchar_func orig_fn = nullptr;

extern "C" void bpftime_agent_main(const gchar *data, gboolean *stay_resident);

// extern "C" int putchar(int c)
// {
// 	if (!orig_fn) {
// 		// if not init, run the bpftime_agent_main to start the client
// 		orig_fn = (putchar_func)dlsym(RTLD_NEXT, "putchar");
// 		printf("new main\n");
// 		int stay_resident = 0;
// 		bpftime_agent_main("", (gboolean *)&stay_resident);
// 	}
// 	return orig_fn(c);
// }
// }
extern "C" int puts(const char *str)
{
	if (!orig_puts_func) {
		// if not init, run the bpftime_agent_main to start the client
		orig_puts_func = (puts_func_t)dlsym(RTLD_NEXT, "puts");
		spdlog::info("Entering new main..");
		int stay_resident = 0;
		bpftime_agent_main("", (gboolean *)&stay_resident);
	}
	return orig_puts_func(str);
}
extern "C" void bpftime_agent_main(const gchar *data, gboolean *stay_resident)
{
	spdlog::cfg::load_env_levels();
	/* We don't want to our library to be unloaded after we return. */
	*stay_resident = TRUE;
	if (!orig_fn) {
		// avoid duplicate init
		orig_fn = (putchar_func)dlsym(RTLD_NEXT, "putchar");
	}
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
	bpftime::setup_syscall_tracer();
	spdlog::info("Loading dynamic library..");
	auto next_handle =
		dlmopen(LM_ID_NEWLM, agent_so, RTLD_NOW | RTLD_LOCAL);
	if (next_handle == nullptr) {
		spdlog::error("Failed to open agent: {}", dlerror());
		exit(1);
	}
	auto entry_func = (void (*)(syscall_hooker_func_t *))dlsym(
		next_handle, "__c_abi_setup_syscall_trace_callback");
	assert(entry_func &&
	       "Malformed agent so, expected symbol __c_abi_setup_syscall_trace_callback");
	syscall_hooker_func_t orig_syscall_hooker_func =
		bpftime::get_call_hook();
	entry_func(&orig_syscall_hooker_func);
	bpftime::set_call_hook(orig_syscall_hooker_func);
	spdlog::info("Transformer exiting..");
}
