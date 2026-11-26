/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#include <catch2/catch_test_macros.hpp>

#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__)) && \
	BPFTIME_HAVE_FRIDA_ATTACH
#include "shm_allocation_test_paths.hpp"
#include "bpftime_agent_ipc.hpp"
#include "bpftime_config.hpp"
#include "bpftime_shm.hpp"
#include "bpftime_shm_internal.hpp"
#include <linux/filter.h>
#include <poll.h>
#include <signal.h>
#include <catch2/generators/catch_generators.hpp>
#include <linux/bpf.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <string>
#include <utility>
#include <sys/wait.h>
#include <unistd.h>

namespace
{

constexpr int MAP_A_FD = 1200;
constexpr int PROG_A_FD = 1201;
constexpr int MAP_B_FD = 1202;
constexpr int PROG_B_FD = 1203;

struct shm_fixture {
	std::string name;
	shm_fixture()
	{
		name = "bpftime-agent-reconcile-test-" +
		       std::to_string(getpid());
		setenv("BPFTIME_GLOBAL_SHM_NAME", name.c_str(), 1);
		bpftime_destroy_global_shm();
		bpftime_initialize_global_shm(
			bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);
		bpftime::runtime_config config;
		config.set_vm_name(DEFAULT_VM_NAME);
		bpftime::shm_holder.global_shared_memory.set_runtime_config(
			std::move(config));
	}
	~shm_fixture()
	{
		bpftime_destroy_global_shm();
		bpftime_initialize_global_shm(
			bpftime::shm_open_type::SHM_NO_CREATE);
		unsetenv("BPFTIME_GLOBAL_SHM_NAME");
	}
};

struct dso_symbol {
	std::string path;
	uint64_t offset = 0;
};

// Runtime offset and resolved path of the DSO symbol from /proc/self/maps.
dso_symbol dso_symbol_offset(const char *lib_path, const char *symbol)
{
	void *handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
	REQUIRE(handle != nullptr);
	void *sym = dlsym(handle, symbol);
	REQUIRE(sym != nullptr);
	const char *slash = strrchr(lib_path, '/');
	std::string base_name = slash ? slash + 1 : lib_path;
	uint64_t lowest = UINT64_MAX;
	std::string mapped_path;
	std::ifstream maps("/proc/self/maps");
	std::string line;
	while (std::getline(maps, line)) {
		if (line.find(base_name) == std::string::npos)
			continue;
		uint64_t start = 0;
		char path_buf[4096] = {};
		if (sscanf(line.c_str(), "%lx-%*x %*4s %*x %*s %*u %4095s",
			   &start, path_buf) >= 1 &&
		    start < lowest) {
			lowest = start;
			if (path_buf[0] == '/')
				mapped_path = path_buf;
		}
	}
	REQUIRE(lowest != UINT64_MAX);
	REQUIRE(!mapped_path.empty());
	uint64_t offset = reinterpret_cast<uint64_t>(sym) - lowest;
	REQUIRE(dlclose(handle) == 0);
	return { std::move(mapped_path), offset };
}

// Increment key 0 once for each probe invocation.
int create_hit_prog(int prog_fd, int map_fd, const char *name)
{
	bpf_insn prog[] = {
		BPF_LD_IMM64_RAW_FULL(BPF_REG_1, BPF_PSEUDO_MAP_FD, 0, 0,
				      map_fd, 0),
		BPF_MOV64_REG(BPF_REG_2, BPF_REG_10),
		BPF_ALU64_IMM(BPF_ADD, BPF_REG_2, -4),
		BPF_ST_MEM(BPF_W, BPF_REG_2, 0, 0),
		BPF_EMIT_CALL(BPF_FUNC_map_lookup_elem),
		BPF_JMP_IMM(BPF_JEQ, BPF_REG_0, 0, 3),
		BPF_LDX_MEM(BPF_DW, BPF_REG_1, BPF_REG_0, 0),
		BPF_ALU64_IMM(BPF_ADD, BPF_REG_1, 1),
		BPF_STX_MEM(BPF_DW, BPF_REG_0, BPF_REG_1, 0),
		BPF_MOV64_IMM(BPF_REG_0, 0),
		BPF_EXIT_INSN(),
	};
	return bpftime_progs_create(
		prog_fd, reinterpret_cast<const ebpf_inst *>(prog),
		sizeof(prog) / sizeof(prog[0]), name, BPF_PROG_TYPE_KPROBE);
}

int create_array_map(int fd, const char *name)
{
	return bpftime_maps_create(
		fd, name,
		bpftime::bpf_map_attr{ .type = BPF_MAP_TYPE_ARRAY,
				       .key_size = sizeof(uint32_t),
				       .value_size = sizeof(uint64_t),
				       .max_ents = 1 });
}

int create_uprobe_link(int prog_fd, const char *lib_path, uint64_t offset)
{
	int perf_fd =
		bpftime_uprobe_create(-1, -1, lib_path, offset, false, 0);
	REQUIRE(perf_fd >= 0);
	bpf_link_create_args args{};
	args.prog_fd = prog_fd;
	args.target_fd = static_cast<uint32_t>(perf_fd);
	args.attach_type = BPFTIME_BPF_PERF_EVENT_ATTACH_TYPE;
	int link_fd = bpftime_link_create(-1, &args);
	REQUIRE(link_fd >= 0);
	return link_fd;
}

uint64_t read_map_value(int fd)
{
	uint32_t key = 0;
	const void *value = bpftime_map_lookup_elem(fd, &key);
	return value == nullptr ? 0 : *static_cast<const uint64_t *>(value);
}

std::string read_line(int fd)
{
	std::string result;
	char ch = 0;
	for (;;) {
		pollfd pfd{ fd, POLLIN, 0 };
		REQUIRE(poll(&pfd, 1, 10000) == 1);
		if (read(fd, &ch, 1) != 1 || ch == '\n')
			break;
		result.push_back(ch);
	}
	return result;
}

} // namespace

TEST_CASE("Live agent reconciles new modules and preserves its session",
	  "[agent][reconcile]")
{
	bool enable_ipc = GENERATE(false, true);
	shm_fixture shm;
	const auto initial_sym = dso_symbol_offset("libc.so.6", "getpgid");
	const auto late_sym = dso_symbol_offset(
		BPFTIME_AGENT_RECONCILE_TARGET_LIB,
		"bpftime_reconcile_target_func");
	REQUIRE(create_array_map(MAP_A_FD, "reconcile_map_a") >= 0);
	REQUIRE(create_hit_prog(PROG_A_FD, MAP_A_FD, "reconcile_prog_a") >= 0);
	REQUIRE(create_uprobe_link(PROG_A_FD, initial_sym.path.c_str(),
				   initial_sym.offset) >= 0);
	REQUIRE(create_array_map(MAP_B_FD, "reconcile_map_b") >= 0);
	REQUIRE(create_hit_prog(PROG_B_FD, MAP_B_FD, "reconcile_prog_b") >= 0);

	int ctrl_pipe[2];
	int out_pipe[2];
	REQUIRE(pipe(ctrl_pipe) == 0);
	REQUIRE(pipe(out_pipe) == 0);
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		close(ctrl_pipe[1]);
		close(out_pipe[0]);
		dup2(out_pipe[1], STDOUT_FILENO);
		close(out_pipe[1]);
		setenv("LD_PRELOAD", BPFTIME_AGENT_LIBRARY, 1);
		if (enable_ipc)
			setenv("BPFTIME_ENABLE_AGENT_IPC", "1", 1);
		else
			unsetenv("BPFTIME_ENABLE_AGENT_IPC");
		unsetenv("BPFTIME_AGENT_DENY_PROCESS");
		unsetenv("BPFTIME_AGENT_ALLOW_PROCESS");
		std::string fd_arg = std::to_string(ctrl_pipe[0]);
		execl(BPFTIME_AGENT_RECONCILE_HELPER,
		      "bpftime-agent-reconcile-helper",
		      BPFTIME_AGENT_RECONCILE_TARGET_LIB, fd_arg.c_str(), nullptr);
		_exit(127);
	}
	close(ctrl_pipe[0]);
	close(out_pipe[1]);
	struct child_cleanup {
		pid_t pid;
		int input, output;
		~child_cleanup()
		{
			if (pid > 0) {
				kill(pid, SIGKILL);
				waitpid(pid, nullptr, 0);
			}
			close(input);
			close(output);
		}
	} cleanup{ child, ctrl_pipe[1], out_pipe[0] };
	std::string ready = read_line(out_pipe[0]);
	if (ready != "ready") {
		int early_status = 0;
		pid_t w = waitpid(child, &early_status, WNOHANG);
		if (w == child)
			cleanup.pid = -1;
		INFO("child waitpid=" << w << " exited=" << WIFEXITED(early_status)
				      << " code=" << WEXITSTATUS(early_status)
				      << " sig=" << WTERMSIG(early_status));
	}
	REQUIRE(ready == "ready");
	REQUIRE(read_map_value(MAP_A_FD) == 1);
	REQUIRE(read_map_value(MAP_B_FD) == 0);
	if (!enable_ipc) {
		std::string ipc_probe;
		REQUIRE_FALSE(bpftime::try_send_agent_ipc(child, "status",
							  &ipc_probe, 100));
	}
	int late_link = create_uprobe_link(
		PROG_B_FD, BPFTIME_AGENT_RECONCILE_TARGET_LIB, late_sym.offset);

	auto step = [&](char command, uint64_t initial_hits, uint64_t late_hits) {
		REQUIRE(write(ctrl_pipe[1], &command, 1) == 1);
		REQUIRE(read_line(out_pipe[0]) == "done");
		REQUIRE(read_map_value(MAP_A_FD) == initial_hits);
		REQUIRE(read_map_value(MAP_B_FD) == late_hits);
	};
	auto control = [&](const char *command) {
		std::string response;
		REQUIRE(bpftime::try_send_agent_ipc(child, command, &response, 1000));
		REQUIRE(response == "ok\n");
	};
	step('l', 2, 1);
	step('l', 3, 2);
	if (enable_ipc) {
		// A second link to a live address is deferred until the old link retires.
		REQUIRE(create_uprobe_link(
				PROG_B_FD, BPFTIME_AGENT_RECONCILE_TARGET_LIB,
				late_sym.offset) >= 0);
		control("reconcile");
		step('h', 4, 3);
		bpftime_close(late_link);
		control("reconcile");
		step('h', 5, 4);
		bpftime_close(PROG_B_FD);
		control("reconcile");
		step('h', 6, 4);
		control("detach");
		step('l', 6, 4);
	}
	char quit = 'q';
	REQUIRE(write(ctrl_pipe[1], &quit, 1) == 1);
	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	cleanup.pid = -1;
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
}
#endif
