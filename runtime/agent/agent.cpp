#include "attach_private_data.hpp"
#include "bpf_attach_ctx.hpp"
#include "bpftime_shm_internal.hpp"
#include "frida_attach_private_data.hpp"
#include "frida_uprobe_attach_impl.hpp"
#include "spdlog/common.h"
#include "syscall_trace_attach_impl.hpp"
#include "syscall_trace_attach_private_data.hpp"
#include <cassert>
#include <ctime>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <ostream>
#include <string_view>
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
using namespace bpftime::attach;
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
	// Register syscall trace impl
	auto syscall_trace_impl = std::make_unique<syscall_trace_attach_impl>();
	syscall_trace_impl->set_original_syscall_function(orig_hooker);
	syscall_trace_impl->set_to_global();
	ctx_holder.ctx.register_attach_impl(
		{ ATTACH_SYSCALL_TRACE }, std::move(syscall_trace_impl),
		[](const std::string_view &sv, int &err) {
			auto priv_data = std::make_unique<attach_private_data>(
				syscall_trace_attach_private_data());
			if (int e = priv_data->initialize_from_string(sv);
			    e < 0) {
				err = e;
				return std::unique_ptr<attach_private_data>();
			}
			return priv_data;
		});
	// Register uprobe attach impl
	ctx_holder.ctx.register_attach_impl(
		{ ATTACH_UPROBE, ATTACH_URETPROBE, ATTACH_UPROBE_OVERRIDE,
		  ATTACH_UREPLACE },
		std::make_unique<attach::frida_attach_impl>(),
		[](const std::string_view &sv, int &err) {
			auto priv_data = std::make_unique<attach_private_data>(
				frida_attach_private_data());
			if (int e = priv_data->initialize_from_string(sv);
			    e < 0) {
				err = e;
				return std::unique_ptr<attach_private_data>();
			}
			return priv_data;
		});
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
