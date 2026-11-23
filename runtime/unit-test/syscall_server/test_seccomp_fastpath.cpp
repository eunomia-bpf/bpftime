/* SPDX-License-Identifier: MIT
 *
 * End-to-end coverage for the rootless seccomp cBPF argument fast-path.
 *
 * The helper process runs bpftime-syscall-server via LD_PRELOAD with
 * BPFTIME_GO_SECCOMP=1 and issues raw x86_64 syscalls (the Go workload
 * path), so each syscall really crosses the installed seccomp filter.
 * It checks that anonymous mmap/munmap, mremap, non-DUPFD fcntls and
 * non-perf ioctls run in-kernel while BPF/RINGBUF mmap, F_DUPFD(_CLOEXEC)
 * on map fds and perf-event ioctls are still delivered to the emulation
 * workers.
 */
#include "catch2/catch_test_macros.hpp"

#if defined(__linux__) && defined(__x86_64__)
#include "shm_allocation_test_paths.hpp"
#include <boost/interprocess/shared_memory_object.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace
{
struct helper_outcome {
	int exit_status = -1;
	std::map<std::string, int> checks;
	int summary_failures = -1;
};

helper_outcome
run_helper()
{
	const std::string shm_name =
		"SeccompFastpathTestShm-" + std::to_string(getpid());
	boost::interprocess::shared_memory_object::remove(shm_name.c_str());

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
		if (setenv("BPFTIME_GO_SECCOMP", "1", 1) != 0 ||
		    setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1) !=
			    0 ||
		    setenv("LD_PRELOAD", BPFTIME_SYSCALL_SERVER_LIBRARY, 1) !=
			    0)
			_exit(126);
		execl(BPFTIME_SECCOMP_FASTPATH_TEST_HELPER,
		      BPFTIME_SECCOMP_FASTPATH_TEST_HELPER, nullptr);
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
	boost::interprocess::shared_memory_object::remove(shm_name.c_str());

	helper_outcome result;
	result.exit_status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
	size_t pos = 0;
	while (pos < output.size()) {
		size_t end = output.find('\n', pos);
		if (end == std::string::npos)
			end = output.size();
		std::string line = output.substr(pos, end - pos);
		pos = end + 1;
		int value = -1;
		char name[128] = {};
		if (sscanf(line.c_str(), "check=%127[^=]=%d", name, &value) ==
		    2)
			result.checks[name] = value;
		int failures = 0;
		if (sscanf(line.c_str(), "SUMMARY failures=%d", &failures) ==
		    1)
			result.summary_failures = failures;
	}
	return result;
}
} // namespace

TEST_CASE("Rootless seccomp cBPF argument fast-path", "[seccomp]")
{
	const helper_outcome outcome = run_helper();

	REQUIRE(outcome.summary_failures == 0);
	REQUIRE(outcome.exit_status == 0);

	// The filter must actually be installed (SECCOMP_MODE_FILTER == 2).
	REQUIRE(outcome.checks.count("seccomp_mode") == 1);
	CHECK(outcome.checks.at("seccomp_mode") == 0);

	// Fast-path: handled by the kernel without a user notification.
	CHECK(outcome.checks.at("anon_mmap_munmap") == 0);
	CHECK(outcome.checks.at("mremap") == 0);
	CHECK(outcome.checks.at("brk_raw") == 0);
	CHECK(outcome.checks.at("brk_malloc") == 0);
	CHECK(outcome.checks.at("fcntl_getfl_setfl") == 0);
	CHECK(outcome.checks.at("ioctl_fionread") == 0);

	// Still trapped and emulated: fake-kernel semantics must survive.
	CHECK(outcome.checks.at("bpf_map_create") == 0);
	CHECK(outcome.checks.at("fcntl_dupfd_cloexec_map") == 0);
	CHECK(outcome.checks.at("mmap_ringbuf_map") == 0);
	CHECK(outcome.checks.at("ioctl_perf_event_enable") == 0);
}
#endif
