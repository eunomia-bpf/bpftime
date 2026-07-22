/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#include "catch2/catch_test_macros.hpp"

#if defined(__linux__)
#include "shm_allocation_test_paths.hpp"

#include <boost/interprocess/shared_memory_object.hpp>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

namespace
{
struct helper_result {
	int status;
	std::string output;
};

helper_result run_preloaded_helper(const char *mode, const char *library,
				   const char *memory_mb = nullptr,
				   const char *max_fd_count = nullptr,
				   const char *agent_so = nullptr)
{
	const std::string shm_name = "bpftime-shm-allocation-test-" +
				     std::string(mode) + "-" +
				     std::to_string(getpid());
	boost::interprocess::shared_memory_object::remove(shm_name.c_str());
	int output_pipe[2];
	REQUIRE(pipe(output_pipe) == 0);

	pid_t pid = fork();
	REQUIRE(pid >= 0);
	if (pid == 0) {
		close(output_pipe[0]);
		if (dup2(output_pipe[1], STDOUT_FILENO) == -1 ||
		    dup2(output_pipe[1], STDERR_FILENO) == -1) {
			_exit(125);
		}
		close(output_pipe[1]);
		if (unsetenv("BPFTIME_LOG_OUTPUT") != 0 ||
		    unsetenv("BPFTIME_SHM_MEMORY_MB") != 0 ||
		    unsetenv("BPFTIME_MAX_FD_COUNT") != 0 ||
		    setenv("HOME", "/proc/bpftime-unwritable-home", 1) != 0 ||
		    setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1) !=
			    0 ||
		    setenv("LD_PRELOAD", library, 1) != 0) {
			_exit(126);
		}
		if (memory_mb != nullptr &&
		    setenv("BPFTIME_SHM_MEMORY_MB", memory_mb, 1) != 0)
			_exit(126);
		if (max_fd_count != nullptr &&
		    setenv("BPFTIME_MAX_FD_COUNT", max_fd_count, 1) != 0)
			_exit(126);
		if (agent_so != nullptr) {
			if (setenv("AGENT_SO", agent_so, 1) != 0)
				_exit(126);
		} else if (unsetenv("AGENT_SO") != 0) {
			_exit(126);
		}
		execl(BPFTIME_SHM_ALLOCATION_TEST_HELPER,
		      BPFTIME_SHM_ALLOCATION_TEST_HELPER, mode, nullptr);
		_exit(127);
	}

	close(output_pipe[1]);
	std::string output;
	std::array<char, 4096> buffer;
	for (;;) {
		ssize_t count =
			read(output_pipe[0], buffer.data(), buffer.size());
		if (count > 0) {
			output.append(buffer.data(),
				      static_cast<size_t>(count));
			continue;
		}
		if (count == -1 && errno == EINTR)
			continue;
		REQUIRE(count == 0);
		break;
	}
	close(output_pipe[0]);

	int status = 0;
	while (waitpid(pid, &status, 0) == -1) {
		REQUIRE(errno == EINTR);
	}
	boost::interprocess::shared_memory_object::remove(shm_name.c_str());
	return { status, std::move(output) };
}
} // namespace

TEST_CASE("Syscall server fails open when startup shared memory is too small",
	  "[allocation][syscall_server]")
{
	auto result = run_preloaded_helper(
		"startup", BPFTIME_SYSCALL_SERVER_LIBRARY, "64", "1048576");
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 100);
	REQUIRE(result.output.empty());
}

TEST_CASE("Syscall server perf mmap reports shared memory exhaustion",
	  "[allocation][syscall_server]")
{
	auto result = run_preloaded_helper(
		"perf-mmap", BPFTIME_SYSCALL_SERVER_LIBRARY, "4", "128");
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 0);
	REQUIRE(result.output.empty());
}

TEST_CASE("Agent initialization failure preserves host exit and stdio",
	  "[preload][agent]")
{
	auto result =
		run_preloaded_helper("passthrough", BPFTIME_AGENT_LIBRARY);
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 23);
	REQUIRE(result.output == "host stdout\nhost stderr\n");
}

#if defined(__x86_64__)
TEST_CASE("Transformer setup failure preserves host exit and stdio",
	  "[preload][transformer]")
{
	auto result =
		run_preloaded_helper("passthrough", BPFTIME_TRANSFORMER_LIBRARY,
				     nullptr, nullptr,
				     "/nonexistent/bpftime-agent.so");
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 23);
	REQUIRE(result.output == "host stdout\nhost stderr\n");
}
#endif
#endif
