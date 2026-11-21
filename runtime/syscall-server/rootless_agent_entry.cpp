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
 * Two arming paths cover every process startup model:
 *  - a dynamic-loader constructor arms the seccomp USER_NOTIF filter for
 *    processes that never run glibc __libc_start_main (notably Go programs,
 *    whose rt0 entry bypasses the glibc startup);
 *  - the __libc_start_main interposer keeps the original path for ordinary
 *    C/C++ programs.
 * An atomic guard makes the two paths mutually idempotent so the filter and
 * its worker threads are installed exactly once.
 */
#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))
#include "seccomp_tracer.hpp"
#include "syscall_server_internal_dispatch.hpp"
#include <atomic>
#include <dlfcn.h>
#include <spdlog/spdlog.h>

using bpftime_main_func_t = int (*)(int, char **, char **);

namespace {
// Set once the seccomp filter has been armed (or arming was attempted), so the
// constructor and the glibc entry interposer cannot install two listeners.
std::atomic<bool> g_rootless_seccomp_armed{false};

void rootless_arm_seccomp()
{
	// Keep the env gate ahead of every side effect: a preload without
	// BPFTIME_GO_SECCOMP must behave exactly as if this library were absent.
	if (getenv("BPFTIME_GO_SECCOMP") == nullptr)
		return;
	bool expected = false;
	if (!g_rootless_seccomp_armed.compare_exchange_strong(expected, true))
		return;

	// This runs across an exported boundary (loader constructor and the
	// __libc_start_main interposer), so no exception may escape: fail open
	// and leave the host running with ordinary syscalls.
	try {
		// Worker threads log through interposed syscalls; a rotating
		// file sink makes a worker's spdlog flush contend with the main
		// thread's stdio lock during fclose and deadlock. Default to
		// console on the rootless path only: an ordinary
		// BPFTIME_LOG_OUTPUT-less preload must keep writing its log
		// file, as the startup tests assert no host stdio.
		if (getenv("BPFTIME_LOG_OUTPUT") == nullptr)
			setenv("BPFTIME_LOG_OUTPUT", "console", 1);

		// Finish the one-time fake-kernel/SHM startup BEFORE installing
		// the filter. The startup path issues whitelisted syscalls;
		// doing it lazily inside a seccomp worker self-deadlocks because
		// the workers block on this same initialization.
		if (bpftime_syscall_server_prewarm() != 0) {
			SPDLOG_ERROR(
				"rootless: syscall server prewarm failed; skipping seccomp");
			return;
		}

		// Install the seccomp USER_NOTIF filter with workers started
		// first.
		bpftime::seccomp::install_seccomp_tracer();
	} catch (...) {
		// Fail open: the filter is optional, the host is not.
	}
}
} // namespace

__attribute__((constructor)) static void rootless_seccomp_constructor()
{
	// The dynamic loader runs preload constructors in every process,
	// including programs whose own _start bypasses glibc __libc_start_main
	// (e.g. the Go runtime rt0 entry), so this is the arming point that
	// covers those runtimes.
	rootless_arm_seccomp();
}

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

	rootless_arm_seccomp();

	return orig(main, argc, argv, init, fini, rtld_fini, stack_end);
}
#endif
