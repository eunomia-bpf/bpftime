/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#include "catch2/catch_test_macros.hpp"

#if defined(__linux__)
#include "shm_allocation_test_paths.hpp"

#include <boost/interprocess/shared_memory_object.hpp>
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

helper_result run_allocation_helper(const char *mode, const char *memory_mb,
				    const char *max_fd_count,
				    const char *log_output,
				    const char *log_level = nullptr)
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
		    unsetenv("SPDLOG_LEVEL") != 0 ||
		    unsetenv("BPFTIME_SHM_MEMORY_MB") != 0 ||
		    unsetenv("BPFTIME_MAX_FD_COUNT") != 0 ||
		    setenv("HOME", "/proc/bpftime-unwritable-home", 1) != 0 ||
		    setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1) !=
			    0 ||
		    setenv("LD_PRELOAD", BPFTIME_SYSCALL_SERVER_LIBRARY, 1) !=
			    0) {
			_exit(126);
		}
		if (memory_mb != nullptr &&
		    setenv("BPFTIME_SHM_MEMORY_MB", memory_mb, 1) != 0)
			_exit(126);
		if (max_fd_count != nullptr &&
		    setenv("BPFTIME_MAX_FD_COUNT", max_fd_count, 1) != 0)
			_exit(126);
		if (log_output != nullptr &&
		    setenv("BPFTIME_LOG_OUTPUT", log_output, 1) != 0)
			_exit(126);
		if (log_level != nullptr &&
		    setenv("SPDLOG_LEVEL", log_level, 1) != 0)
			_exit(126);
		execl(BPFTIME_SHM_ALLOCATION_TEST_HELPER,
		      BPFTIME_SHM_ALLOCATION_TEST_HELPER, mode, nullptr);
		_exit(127);
	}

	close(output_pipe[1]);
	std::string output;
	char buffer[4096];
	for (;;) {
		ssize_t count = read(output_pipe[0], buffer, sizeof(buffer));
		if (count > 0) {
			output.append(buffer, static_cast<size_t>(count));
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

TEST_CASE("Syscall server exits cleanly when startup shared memory is too small",
	  "[allocation][syscall_server]")
{
	auto result =
		run_allocation_helper("startup", "64", "1048576", "console");
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 1);
}

TEST_CASE("Syscall server perf mmap reports shared memory exhaustion",
	  "[allocation][syscall_server]")
{
	auto result = run_allocation_helper("perf-mmap", "4", "128", "console");
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 0);
}

TEST_CASE("Syscall server keeps logger sink failures off host stdio",
	  "[allocation][syscall_server][logging]")
{
	auto result =
		run_allocation_helper("startup", "64", "1048576", "/dev/full");
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 1);
	REQUIRE(result.output.empty());
}

TEST_CASE("Syscall server keeps default logging off host stdio",
	  "[allocation][syscall_server][logging]")
{
	auto result =
		run_allocation_helper("startup", "64", "1048576", nullptr);
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 1);
	REQUIRE(result.output.empty());
}

TEST_CASE("Syscall server preserves the console logger level name",
	  "[allocation][syscall_server][logging]")
{
	auto result = run_allocation_helper("startup", "64", "1048576",
					    "console", "stderr=off");
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 1);
	REQUIRE(result.output.empty());
}
#endif
