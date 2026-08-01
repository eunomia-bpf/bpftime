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

namespace
{
int run_allocation_helper(const char *mode, const char *memory_mb,
			  const char *max_fd_count)
{
	const std::string shm_name =
		"bpftime-shm-allocation-test-" + std::string(mode) + "-" +
		std::to_string(getpid());
	boost::interprocess::shared_memory_object::remove(shm_name.c_str());

	pid_t pid = fork();
	REQUIRE(pid >= 0);
	if (pid == 0) {
		if (setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1) != 0 ||
		    setenv("BPFTIME_LOG_OUTPUT", "console", 1) != 0 ||
		    setenv("BPFTIME_SHM_MEMORY_MB", memory_mb, 1) != 0 ||
		    setenv("BPFTIME_MAX_FD_COUNT", max_fd_count, 1) != 0 ||
		    setenv("LD_PRELOAD", BPFTIME_SYSCALL_SERVER_LIBRARY, 1) != 0) {
			_exit(126);
		}
		execl(BPFTIME_SHM_ALLOCATION_TEST_HELPER,
		      BPFTIME_SHM_ALLOCATION_TEST_HELPER, mode, nullptr);
		_exit(127);
	}

	int status = 0;
	while (waitpid(pid, &status, 0) == -1) {
		REQUIRE(errno == EINTR);
	}
	boost::interprocess::shared_memory_object::remove(shm_name.c_str());
	return status;
}
} // namespace

TEST_CASE("Syscall server exits cleanly when startup shared memory is too small",
	  "[allocation][syscall_server]")
{
	int status = run_allocation_helper("startup", "64", "1048576");
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 1);
}

TEST_CASE("Syscall server perf mmap reports shared memory exhaustion",
	  "[allocation][syscall_server]")
{
	int status = run_allocation_helper("perf-mmap", "4", "128");
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
}
#endif
