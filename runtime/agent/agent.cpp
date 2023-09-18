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
	res = ctx.init_attach_ctx_from_handlers(config);
	if (res != 0) {
		g_print("Failed to init attach ctx\n");
		return;
	}
	g_print("Successfully attached\n");

	// don't free ctx here
	return;
}
syscall_hooker_func_t orig_hooker;

int64_t test_hooker(int64_t sys_nr, int64_t arg1, int64_t arg2, int64_t arg3,
		    int64_t arg4, int64_t arg5, int64_t arg6)
{
	std::cout << "SYS " << sys_nr << std::endl;
	return orig_hooker(sys_nr, arg1, arg2, arg3, arg4, arg5, arg6);
}

extern "C" void
__c_abi_setup_syscall_trace_callback(syscall_hooker_func_t *hooker)
{
	orig_hooker = *hooker;
	*hooker = &test_hooker;
	gboolean val;
	bpftime_agent_main("", &val);
}
