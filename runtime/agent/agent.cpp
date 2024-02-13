#include "bpf_attach_ctx.hpp"
#include "bpftime_shm_internal.hpp"
#include "spdlog/common.h"
#include "syscall_trace_attach_impl.hpp"
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

syscall_hooker_func_t orig_hooker;

extern "C" void bpftime_agent_main(const gchar *data, gboolean *stay_resident);

extern "C" int bpftime_hooked_main(int argc, char **argv, char **envp)
{
	int stay_resident = 0;

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
	SPDLOG_INFO("Entering bpftime agent");
	orig_main_func = main;
	using this_func_t = decltype(&__libc_start_main);
	this_func_t orig = (this_func_t)dlsym(RTLD_NEXT, "__libc_start_main");

	return orig(bpftime_hooked_main, argc, argv, init, fini, rtld_fini,
		    stack_end);
}

extern "C" void bpftime_agent_main(const gchar *data, gboolean *stay_resident)
{
	spdlog::cfg::load_env_levels();
	bpftime_initialize_global_shm(shm_open_type::SHM_OPEN_ONLY);
	ctx_holder.init();
	auto &impl = dynamic_cast<attach::syscall_trace_attach_impl &>(
		ctx_holder.ctx.get_syscall_attach_impl());

	impl.set_original_syscall_function(orig_hooker);
	impl.set_to_global();
	// ctx_holder.ctx.set_orig_syscall_func(orig_hooker);

	SPDLOG_INFO("Initializing agent..");
	spdlog::set_pattern("[%Y-%m-%d %H:%M:%S][%^%l%$][%t] %v");
	/* We don't want to our library to be unloaded after we return. */
	*stay_resident = TRUE;

	int res = 1;
	setenv("BPFTIME_USED", "1", 0);
	SPDLOG_DEBUG("Set environment variable BPFTIME_USED");
	res = ctx_holder.ctx.init_attach_ctx_from_handlers(
		bpftime_get_agent_config());
	if (res != 0) {
		SPDLOG_ERROR("Failed to initialize attach context");
		return;
	}
	SPDLOG_INFO("Attach successfully");
	// don't free ctx here
	return;
}

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
