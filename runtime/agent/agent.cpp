#include "bpf_attach_ctx.hpp"
#include "bpftime_shm_internal.hpp"
#include "spdlog/common.h"
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
#include <inttypes.h>
#include <dlfcn.h>
#include "bpftime.hpp"
#include "bpftime_shm.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>
using namespace bpftime;

const shm_open_type bpftime::global_shm_open_type = shm_open_type::SHM_CLIENT;

using main_func_t = int (*)(int, char **, char **);

static main_func_t orig_main_func = nullptr;

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

extern "C" void bpftime_agent_main(const gchar *data, gboolean *stay_resident);

extern "C" int bpftime_hooked_main(int argc, char **argv, char **envp)
{
	int stay_resident = 0;
	spdlog::cfg::load_env_levels();
	bpftime_initialize_global_shm();
	ctx_holder.init();
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
	spdlog::info("Initializing agent..");
	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S][%^%l%$][%t] %v");
	/* We don't want to our library to be unloaded after we return. */
	*stay_resident = TRUE;

	int res = 1;

	res = ctx_holder.ctx.init_attach_ctx_from_handlers(
		bpftime_get_agent_config());
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
	// spdlog::info(
	// "Calling syscall callback: sys_nr {}, args {} {} {} {} {} {}",
	// sys_nr, arg1, arg2, arg3, arg4, arg5, arg6);
	return ctx_holder.ctx.run_syscall_hooker(sys_nr, arg1, arg2, arg3, arg4,
						 arg5, arg6);
}

extern "C" void
__c_abi_setup_syscall_trace_callback(syscall_hooker_func_t *hooker)
{
	orig_hooker = *hooker;
	*hooker = &syscall_callback;
	bpftime_initialize_global_shm();
	ctx_holder.init();
	ctx_holder.ctx.set_orig_syscall_func(orig_hooker);
	gboolean val;
	bpftime_agent_main("", &val);
	spdlog::info("Agent syscall trace setup exiting..");
}
