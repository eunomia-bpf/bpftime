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
#include "text_segment_transformer.hpp"
using namespace bpftime;

const shm_open_type bpftime::global_shm_open_type = shm_open_type::SHM_CLIENT;

typedef int (*putchar_func)(int c);

// using putchar_func as a flag to indicate whether the agent has been init
static putchar_func orig_fn = nullptr;

void bpftime_agent_main(const gchar *data, gboolean *stay_resident);

extern "C" int putchar(int c)
{
	if (!orig_fn) {
		// if not init, run the bpftime_agent_main to start the client
		orig_fn = (putchar_func)dlsym(RTLD_NEXT, "putchar");
		printf("new main\n");
		int stay_resident = 0;
		bpftime_agent_main("", (gboolean *)&stay_resident);
	}
	return orig_fn(c);
}

static bpf_attach_ctx ctx;

void bpftime_agent_main(const gchar *data, gboolean *stay_resident)
{
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
	if (ctx.check_exist_syscall_trace_program()) {
		auto self_pid = getpid();
		if (!ctx.check_syscall_trace_setup(self_pid)) {
			std::cout << "Setup userspace syscall tracer"
				  << std::endl;

			bpftime::setup_syscall_tracer();
			ctx.set_syscall_trace_setup(self_pid, true);

			auto so_name = getenv("LD_PRELOAD");
			assert(so_name &&
			       "Currently, only client injected by LD_PRELOAD can use syscall traces");

			// Here, we should load another shared object, which
			// will be in
			// a separate namespace

			auto next_handle = dlmopen(LM_ID_NEWLM, so_name,
						   RTLD_NOW | RTLD_LOCAL);
			auto entry_func = (void (*)(void))dlsym(
				next_handle, "__c_abi_bpftime_agent_main");
			entry_func();
			ctx.set_syscall_trace_setup(self_pid, false);
			return;
		} else {
			std::cout
				<< "Userspace syscall tracer already setup. Entering bpftime.."
				<< std::endl;
		}
	}
	res = ctx.init_attach_ctx_from_handlers(config);
	if (res != 0) {
		g_print("Failed to init attach ctx\n");
		return;
	}
	g_print("Successfully attached\n");

	// don't free ctx here
	return;
}

extern "C" void __c_abi_bpftime_agent_main()
{
	int stay_resident = 0;
	bpftime_agent_main("", (gboolean *)&stay_resident);
}
