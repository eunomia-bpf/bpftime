/* SPDX-License-Identifier: MIT
 *
 * Rootless entry point compiled into bpftime-syscall-server.
 *
 * ARCHITECTURE INVARIANT — DO NOT BREAK:
 * On the rootless path the collector process loads ONLY this shared object.
 * It must not dlopen agent.so or reach any attach/ component
 * (frida/capstone/text-segment transformer). bpf map/prog emulation and the
 * eBPF VM are self-contained in the syscall-server; agent.so belongs
 * exclusively to the observed workload process (a separate worker-lib build
 * opening the same SHM). The in-process uprobe text-segment mode keeps using
 * the transformer and is untouched.
 *
 * We interpose __libc_start_main (a normal libc symbol, available after the
 * loader has initialized pthread) rather than a library constructor: the
 * seccomp worker threads must be fully startable before the filter is
 * installed, which cannot be guaranteed at constructor time.
 */
#if defined(__linux__) && defined(__x86_64__)
#include "seccomp_tracer.hpp"
#include "syscall_server_internal_dispatch.hpp"
#include <dlfcn.h>
#include <spdlog/spdlog.h>

using bpftime_main_func_t = int (*)(int, char **, char **);

extern "C" int __libc_start_main(bpftime_main_func_t main, int argc,
				 char **argv, int (*init)(int, char **, char **),
				 void (*fini)(void), void (*rtld_fini)(void),
				 void *stack_end)
{
	auto *orig = reinterpret_cast<int (*)(bpftime_main_func_t, int, char **,
					      int (*)(int, char **, char **),
					      void (*)(void), void (*)(void),
					      void *)>(
		dlsym(RTLD_NEXT, "__libc_start_main"));

	// Worker threads log through interposed syscalls; a rotating file sink
	// makes a worker's spdlog flush contend with the main thread's stdio
	// lock during fclose and deadlock. Default to console unless the caller
	// chose an explicit target (the omni bundle already sets console).
	if (getenv("BPFTIME_LOG_OUTPUT") == nullptr)
		setenv("BPFTIME_LOG_OUTPUT", "console", 1);

	// Finish the one-time fake-kernel/SHM startup BEFORE installing the
	// filter. The startup path issues whitelisted syscalls; doing it lazily
	// inside a seccomp worker self-deadlocks because the workers block on
	// this same initialization. Non-fatal: on failure skip the filter and
	// leave the host running with ordinary syscalls (fail open).
	if (getenv("BPFTIME_GO_SECCOMP") != nullptr &&
	    bpftime_syscall_server_prewarm() != 0) {
		SPDLOG_ERROR(
			"rootless: syscall server prewarm failed; skipping seccomp");
		return orig(main, argc, argv, init, fini, rtld_fini, stack_end);
	}

	// Install the seccomp USER_NOTIF filter with workers started first.
	// No-op unless BPFTIME_GO_SECCOMP is set, preserving the env gate.
	bpftime::seccomp::install_seccomp_tracer();

	return orig(main, argc, argv, init, fini, rtld_fini, stack_end);
}
#endif
