#include <cassert>
#include <ctime>
#include <fcntl.h>
#include <iostream>
#include <ostream>
#include <unistd.h>
#include <frida-gum.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>
#include <inttypes.h>
#include <dlfcn.h>
#include "bpftime.hpp"
#include "bpftime_shm.hpp"
#include <spdlog/spdlog.h>
using namespace bpftime;

const shm_open_type bpftime::global_shm_open_type = shm_open_type::SHM_CLIENT;

using putchar_func = int (*)(int);
using puts_func_t = int (*)(const char *);

static puts_func_t orig_puts_func = nullptr;

// using putchar_func as a flag to indicate whether the agent has been init
static putchar_func orig_fn = nullptr;

void bpftime_agent_main(const gchar *data, gboolean *stay_resident);

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
extern "C" int puts(const char *str)
{
	if (!orig_puts_func) {
		// if not init, run the bpftime_agent_main to start the client
		orig_puts_func = (puts_func_t)dlsym(RTLD_NEXT, "puts");
		spdlog::info("Entering new main function");
		int stay_resident = 0;
		bpftime_agent_main("", (gboolean *)&stay_resident);
	}
	return orig_puts_func(str);
}
static bpf_attach_ctx ctx;

void bpftime_agent_main(const gchar *data, gboolean *stay_resident)
{
	// spdlog::info("Initializing agent..");
	std::cout<<"Initializing agent..."<<std::endl;
	/* We don't want to our library to be unloaded after we return. */
	*stay_resident = TRUE;
	if (!orig_fn) {
		// avoid duplicate init
		orig_fn = (putchar_func)dlsym(RTLD_NEXT, "putchar");
	}

	int res = 1;

	agent_config config;
	config.enable_ffi_helper_group = true;
	config.enable_shm_maps_helper_group = true;
	res = ctx.init_attach_ctx_from_handlers(config);
	if (res != 0) {
		spdlog::error("Failed to initialize attach context");
		return;
	}
	spdlog::info("Attach successfully");

	// don't free ctx here
	return;
}
syscall_hooker_func_t orig_hooker;

int64_t syscall_callback(int64_t sys_nr, int64_t arg1, int64_t arg2,
			 int64_t arg3, int64_t arg4, int64_t arg5, int64_t arg6)
{
	return ctx.run_syscall_hooker(sys_nr, arg1, arg2, arg3, arg4, arg5,
				      arg6);
}

extern "C" void
__c_abi_setup_syscall_trace_callback(syscall_hooker_func_t *hooker)
{
	orig_hooker = *hooker;
	*hooker = &syscall_callback;
	ctx.set_orig_syscall_func(orig_hooker);
	gboolean val;
	bpftime_agent_main("", &val);
}
