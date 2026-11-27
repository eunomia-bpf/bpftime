// SPDX-License-Identifier: MIT
// Darwin preload-constructor test driver.
//
// It creates the global SHM segment, an array map and a uprobe program, then
// spawns the victim with POSIX_SPAWN_START_SUSPENDED so dyld (and the injected
// agent constructor) does not run until the uprobe/link handlers are already
// present. The child is resumed with SIGCONT, the agent attaches in-process,
// and after the child exits the driver reads the shared map and asserts that
// every target-function call was observed.
#include "bpf_attach_ctx.hpp"
#include "bpftime_shm.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <crt_externs.h>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <signal.h>
#include <spawn.h>
#include <string>
#include <sys/wait.h>

using namespace bpftime;

namespace {
int fail(const std::string &msg)
{
	std::cerr << "PRELOAD-TEST-FAIL: " << msg << "\n";
	return 1;
}
} // namespace

int main(int argc, char **argv)
{
	if (argc != 5) {
		std::cerr << "usage: " << argv[0]
			  << " <agent.dylib> <victim> <target.dylib> <iters>\n";
		return 2;
	}
	const char *agent_dylib = argv[1];
	const char *victim_exe = argv[2];
	const char *target_dylib = argv[3];
	const uint32_t iters = static_cast<uint32_t>(std::stoul(argv[4]));

	const std::string shm_name =
		"bpftime-preload-test-" + std::to_string(getpid()) + "-" +
		std::to_string(std::chrono::steady_clock::now()
				       .time_since_epoch()
				       .count());
	if (setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1) != 0)
		return fail("setenv shm name");
	// SHM creation happens inside the try block below so an allocation
	// failure is covered by the scope-bound cleanup guard.

	int map_fd = -1, prog_fd = -1, event_fd = -1, link_fd = -1;
	pid_t child = -1;
	bool child_reaped = false;
	auto cleanup = [&]() {
		for (int fd : { link_fd, event_fd, prog_fd, map_fd })
			if (fd >= 0) {
				bpftime_close(fd);
				::close(fd);
			}
		// A failure after POSIX_SPAWN_START_SUSPENDED leaves the child
		// stopped; terminate and reap it so no zombie or stopped process
		// survives a failed assertion.
		if (child > 0 && !child_reaped) {
			(void)kill(child, SIGKILL);
			// Block until the kernel confirms termination; a single
			// WNOHANG can race the SIGKILL delivery and leak the
			// child. Retry on EINTR so the reaped state is accurate.
			int st = 0;
			while (waitpid(child, &st, 0) < 0 && errno == EINTR) {
			}
			child_reaped = true;
		}
		if (child > 0) {
			std::filesystem::remove(
				"/tmp/bpftime-maps-" +
				std::to_string(child) + ".txt");
			const char *tmp = std::getenv("TMPDIR");
			std::string dir = (tmp && *tmp) ? tmp : "/tmp/";
			if (dir.back() != '/')
				dir.push_back('/');
			std::filesystem::remove(dir + "bpftime-agent-" +
						std::to_string(child) + ".sock");
		}
		bpftime_destroy_global_shm();
		bpftime_remove_global_shm();
};
	// Scope-bound so every early return and the exception path run cleanup,
	// including failures that occur while the spawned child is suspended.
	struct cleanup_guard {
		decltype(cleanup) &fn;
		~cleanup_guard()
		{
			try {
				fn();
			} catch (...) {
			}
		}
	} guard{ cleanup };

	try {
		// Inside the guarded scope: if construction throws, the guard's
		// cleanup removes the partially created SHM segment.
		bpftime_remove_global_shm();
		bpftime_initialize_global_shm(shm_open_type::SHM_CREATE_ONLY);
		runtime_config config{};
		config.set_vm_name("ubpf");
		config.jit_enabled = false;
		config.enable_kernel_helper_group = false;
		config.enable_ufunc_helper_group = false;
		config.enable_shm_maps_helper_group = true;
		bpftime_set_runtime_config(runtime_config(config));

		// One u64 slot: the uprobe increments it on every entry.
		bpf_map_attr attr{};
		attr.type = static_cast<uint32_t>(bpf_map_type::BPF_MAP_TYPE_ARRAY);
		attr.key_size = sizeof(uint32_t);
		attr.value_size = sizeof(uint64_t);
		attr.max_ents = 1;
		map_fd = bpftime_maps_create(-1, "preload_counter", attr);
		if (map_fd < 0)
			return fail("map creation failed");
		const uint32_t key = 0;
		const uint64_t zero = 0;
		if (bpftime_map_update_elem(map_fd, &key, &zero, 0) != 0)
			return fail("map zero init failed");

		// Portable counter: no pt_regs access, so the bytecode is
		// identical on arm64 and x86_64.
		const ebpf_inst insns[] = {
			{ EBPF_OP_STW, 10, 0, -4, 0 }, // stack key = 0
			{ EBPF_OP_LDDW, 1, 1, 0, map_fd }, { 0, 0, 0, 0, 0 },
			{ EBPF_OP_MOV64_REG, 2, 10, 0, 0 },
			{ EBPF_OP_ADD64_IMM, 2, 0, 0, -4 },
			{ EBPF_OP_CALL, 0, 0, 0, 1 }, // bpf_map_lookup_elem
			{ EBPF_OP_JEQ_IMM, 0, 0, 3, 0 }, // null -> exit
			{ EBPF_OP_LDXDW, 1, 0, 0, 0 },
			{ EBPF_OP_ADD64_IMM, 1, 0, 0, 1 },
			{ EBPF_OP_STXDW, 0, 1, 0, 0 },
			{ EBPF_OP_MOV64_IMM, 0, 0, 0, 0 },
			{ EBPF_OP_EXIT, 0, 0, 0, 0 },
		};
		prog_fd = bpftime_progs_create(
			-1, insns, std::size(insns), "preload_on_entry",
			static_cast<int>(bpf_prog_type::BPF_PROG_TYPE_KPROBE));
		if (prog_fd < 0)
			return fail("prog creation failed");

		// Resolve the module-relative symbol offset the same way the
		// in-process example does; the slide differs per process but
		// the offset is stable.
		void *handle = dlopen(target_dylib, RTLD_NOW | RTLD_LOCAL);
		if (!handle)
			return fail(std::string("dlopen target: ") + dlerror());
		void *sym = dlsym(handle, "bpftime_macos_demo_compute");
		Dl_info info{};
		if (!sym || dladdr(sym, &info) == 0 || !info.dli_fbase)
			return fail("resolve target symbol");
		const uint64_t offset = reinterpret_cast<uintptr_t>(sym) -
				       reinterpret_cast<uintptr_t>(info.dli_fbase);

		// Spawn suspended: dyld and the agent constructor stay frozen
		// until the handlers below are in place and we send SIGCONT.
		posix_spawnattr_t spawn_attr;
		posix_spawnattr_init(&spawn_attr);
		posix_spawnattr_setflags(&spawn_attr, POSIX_SPAWN_START_SUSPENDED);
		std::string iters_str = std::to_string(iters);
		char *const child_argv[] = {
			const_cast<char *>(victim_exe),
			const_cast<char *>(iters_str.c_str()), nullptr
		};
		std::string insert = std::string("DYLD_INSERT_LIBRARIES=") +
				     agent_dylib;
		std::string shm_env =
			"BPFTIME_GLOBAL_SHM_NAME=" + shm_name;
		char **environ = *_NSGetEnviron();
		size_t env_count = 0;
		while (environ[env_count])
			++env_count;
		std::vector<char *> child_env;
		child_env.reserve(env_count + 3);
		for (size_t i = 0; i < env_count; ++i)
			child_env.push_back(environ[i]);
		child_env.push_back(insert.data());
		child_env.push_back(shm_env.data());
		child_env.push_back(nullptr);

		if (posix_spawn(&child, victim_exe, nullptr, &spawn_attr,
				child_argv, child_env.data()) != 0)
			return fail("posix_spawn failed");
		posix_spawnattr_destroy(&spawn_attr);

		event_fd = bpftime_uprobe_create(-1, static_cast<int>(child),
						 target_dylib, offset, false, 0);
		if (event_fd < 0)
			return fail("uprobe create failed");
		bpf_link_create_args link_args{};
		link_args.prog_fd = prog_fd;
		link_args.target_fd = event_fd;
		link_args.attach_type = BPFTIME_BPF_PERF_EVENT_ATTACH_TYPE;
		link_fd = bpftime_link_create(-1, &link_args);
		if (link_fd < 0)
			return fail("link create failed");

		if (kill(child, SIGCONT) != 0)
			return fail("SIGCONT failed");
		int status = 0;
		if (waitpid(child, &status, 0) < 0)
			return fail("waitpid failed");
		child_reaped = true;
		if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
			return fail("victim did not exit cleanly");

		const uint64_t *observed =
			static_cast<const uint64_t *>(
				bpftime_map_lookup_elem(map_fd, &key));
		if (!observed)
			return fail("map lookup after run failed");
		std::cout << "preload test: expected=" << iters
			  << " observed=" << *observed << "\n";
		if (*observed != iters)
			return fail("uprobe call count mismatch");
		std::cout << "PRELOAD-TEST-PASS\n";
		return 0;
	} catch (const std::exception &ex) {
		return fail(std::string("exception: ") + ex.what());
	}
}
