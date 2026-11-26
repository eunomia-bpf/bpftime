/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 *
 * End-to-end coverage for the rootless seccomp constructor in
 * bpftime-syscall-server. The helper binaries are launched through a real
 * LD_PRELOAD of the built shared object:
 *
 *  - a custom-_start helper never runs glibc __libc_start_main, so only the
 *    dynamic-loader constructor can arm the seccomp USER_NOTIF filter;
 *  - an ordinary glibc helper exercises both the constructor and the
 *    __libc_start_main interposer, proving the atomic guard arms once;
 *  - without BPFTIME_GO_SECCOMP the preload stays side-effect free.
 */
#include "catch2/catch_test_macros.hpp"

#if defined(__linux__) && defined(__x86_64__)
#include "shm_allocation_test_paths.hpp"

#include <cerrno>
#include <cstdlib>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace
{
struct helper_run {
	int status = -1;
	// Non-zero only when the helper was killed by a signal.
	int term_signal = 0;
	std::string output;
};

int
count_occurrences(const std::string &haystack, const std::string &needle)
{
	int count = 0;
	size_t pos = 0;
	while ((pos = haystack.find(needle, pos)) != std::string::npos) {
		count++;
		pos += needle.size();
	}
	return count;
}

// Returns the value of the Seccomp: field reported by the helper, or -1.
int
parse_seccomp_mode(const std::string &output)
{
	const std::string marker = "Seccomp:";
	size_t pos = output.find(marker);
	if (pos == std::string::npos)
		return -1;
	pos += marker.size();
	while (pos < output.size() &&
	       (output[pos] == ' ' || output[pos] == '\t'))
		pos++;
	if (pos >= output.size() || output[pos] < '0' || output[pos] > '9')
		return -1;
	return std::atoi(output.c_str() + pos);
}

helper_run
run_start_helper(const char *helper_path, bool preload, bool enable_go_seccomp)
{
	const std::string shm_name = "bpftime-rootless-constructor-test-" +
				     std::to_string(getpid());
	const std::string shm_file = "/dev/shm/" + shm_name;
	::unlink(shm_file.c_str());

	int output_pipe[2];
	REQUIRE(pipe(output_pipe) == 0);
	pid_t pid = fork();
	REQUIRE(pid >= 0);
	if (pid == 0) {
		close(output_pipe[0]);
		if (dup2(output_pipe[1], STDOUT_FILENO) == -1 ||
		    dup2(output_pipe[1], STDERR_FILENO) == -1)
			_exit(125);
		close(output_pipe[1]);
		unsetenv("BPFTIME_LOG_OUTPUT");
		unsetenv("SPDLOG_LEVEL");
		if (setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1) !=
		    0)
			_exit(126);
		if (preload &&
		    setenv("LD_PRELOAD", BPFTIME_SYSCALL_SERVER_LIBRARY, 1) !=
			    0)
			_exit(126);
		if (!preload)
			unsetenv("LD_PRELOAD");
		if (enable_go_seccomp &&
		    setenv("BPFTIME_GO_SECCOMP", "1", 1) != 0)
			_exit(126);
		if (!enable_go_seccomp)
			unsetenv("BPFTIME_GO_SECCOMP");
		execl(helper_path, helper_path, nullptr);
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
	while (waitpid(pid, &status, 0) == -1)
		REQUIRE(errno == EINTR);
	::unlink(shm_file.c_str());
	return { status, WIFSIGNALED(status) ? WTERMSIG(status) : 0,
		 std::move(output) };
}
} // namespace

TEST_CASE("Custom _start helper runs without glibc __libc_start_main",
	  "[rootless][seccomp][constructor]")
{
	auto result =
		run_start_helper(BPFTIME_ROOTLESS_CUSTOM_START_HELPER, false,
				 false);
	REQUIRE(WIFEXITED(result.status));
	REQUIRE(WEXITSTATUS(result.status) == 0);
	REQUIRE(parse_seccomp_mode(result.output) >= 0);
}

TEST_CASE("Loader constructor arms seccomp in a process bypassing "
	  "__libc_start_main",
	  "[rootless][seccomp][constructor]")
{
	auto baseline =
		run_start_helper(BPFTIME_ROOTLESS_CUSTOM_START_HELPER, false,
				 false);
	REQUIRE(WIFEXITED(baseline.status));
	const int baseline_mode = parse_seccomp_mode(baseline.output);
	REQUIRE(baseline_mode >= 0);

	auto armed = run_start_helper(BPFTIME_ROOTLESS_CUSTOM_START_HELPER,
				      true, true);
	REQUIRE(WIFEXITED(armed.status));
	REQUIRE(WEXITSTATUS(armed.status) == 0);
	REQUIRE(parse_seccomp_mode(armed.output) == 2);
	REQUIRE(armed.output.find("Using rootless seccomp notification") !=
		std::string::npos);
	// On an already-confined host the baseline mode may already be 2; the
	// mode-2 and log-marker assertions above still prove the preload
	// installed its own listener. CI runs the job privileged/unconfined.
	if (baseline_mode == 2)
		WARN("outer seccomp filter already present; baseline delta "
		     "not asserted");
	else
		REQUIRE(baseline_mode != 2);
}

TEST_CASE("Preload without BPFTIME_GO_SECCOMP stays side-effect free",
	  "[rootless][seccomp][constructor]")
{
	auto baseline =
		run_start_helper(BPFTIME_ROOTLESS_CUSTOM_START_HELPER, false,
				 false);
	auto quiet = run_start_helper(BPFTIME_ROOTLESS_CUSTOM_START_HELPER,
				      true, false);
	REQUIRE(WIFEXITED(quiet.status));
	REQUIRE(WEXITSTATUS(quiet.status) == 0);
	REQUIRE(parse_seccomp_mode(quiet.output) ==
		parse_seccomp_mode(baseline.output));
	REQUIRE(quiet.output.find("bpftime") == std::string::npos);
	REQUIRE(quiet.output.find("seccomp notification") ==
		std::string::npos);
}

TEST_CASE("Constructor and __libc_start_main arm the filter exactly once",
	  "[rootless][seccomp][constructor]")
{
	auto armed = run_start_helper(BPFTIME_ROOTLESS_NORMAL_START_HELPER,
				      true, true);
	INFO("helper killed by signal " << armed.term_signal
	     << ", output: " << armed.output);
	REQUIRE(WIFEXITED(armed.status));
	REQUIRE(WEXITSTATUS(armed.status) == 0);
	REQUIRE(parse_seccomp_mode(armed.output) == 2);
	REQUIRE(count_occurrences(armed.output,
				  "Using rootless seccomp notification") == 1);
}
#endif
