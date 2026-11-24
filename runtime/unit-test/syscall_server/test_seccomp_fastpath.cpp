/* SPDX-License-Identifier: MIT
 *
 * End-to-end coverage for the rootless seccomp cBPF argument fast-path.
 * The helper process runs bpftime-syscall-server via LD_PRELOAD with
 * BPFTIME_GO_SECCOMP=1 and issues raw x86_64 syscalls (the Go workload
 * path), so each syscall really crosses the installed seccomp filter.
 * It checks that anonymous mmap/munmap, mremap, non-DUPFD fcntls and
 * non-perf ioctls run in-kernel while BPF/RINGBUF mmap, F_DUPFD(_CLOEXEC)
 * on map fds and perf-event ioctls are still delivered to the emulation
 * workers.
 *
 * The child sets BPFTIME_SHM_MEMORY_MB=2: the syscall server otherwise
 * creates its default 50 MiB global shared memory segment, which a small
 * /dev/shm (64 MiB in the CI containers, largely used by earlier tests in
 * the same binary) cannot always fit; the SIGBUS would then be charged to
 * this test instead of the /dev/shm pressure that caused it.
 */
#include "catch2/catch_test_macros.hpp"

#if defined(__linux__) && defined(__x86_64__)
#include "shm_allocation_test_paths.hpp"
#include <boost/interprocess/shared_memory_object.hpp>
#include <sys/statvfs.h>
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
	// RIP and library base captured by the helper's crash handler, as
	// decimal or hex strings; empty when the helper exited normally.
	std::string crash_rip;
	std::string crash_base;
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
		    setenv("BPFTIME_SHM_MEMORY_MB", "2", 1) != 0 ||
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
	if (WIFSIGNALED(status)) {
		// The helper's stdout is fully buffered into the pipe, so a
		// death by signal loses everything it printed. Expose the
		// signal through the parsed output instead: the checks below
		// then fail with the killing signal in the expansion.
		result.checks["helper_signal"] = WTERMSIG(status);
	}
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
		// The crash handler's line:
		// crash_sig=<n> crash_rip=0x.. crash_base=0x..
		char crash_rip[32] = {};
		char crash_base[32] = {};
		if (sscanf(line.c_str(),
			   "crash_sig=%*d crash_rip=%31s crash_base=%31s",
			   crash_rip, crash_base) == 2) {
			result.crash_rip = crash_rip;
			result.crash_base = crash_base;
		}
	}
	return result;
}
} // namespace

TEST_CASE("Rootless seccomp cBPF argument fast-path", "[seccomp]")
{
	// The helper's server creates its 2 MiB shared-memory segment before
	// main() runs, so a SIGBUS from a nearly full /dev/shm surfaces as an
	// opaque exit_status == -1. Skip when the mount cannot hold the
	// segment: the whole-suite run leaves several static segments behind,
	// and sizing the mount is the CI workflow's job, not this test's.
	struct statvfs stat;
	if (statvfs("/dev/shm", &stat) == 0 &&
	    stat.f_bavail * stat.f_bsize < 8ULL * 1024 * 1024) {
		SKIP("free /dev/shm below the 8 MiB the helper needs");
	}
	const helper_outcome outcome = run_helper();

	REQUIRE(outcome.summary_failures == 0);
	if (outcome.checks.count("helper_signal") == 1) {
		// Name the killing signal and every check the helper
		// completed before dying: the expansion then shows where in
		// the sequence the crash happened, not only the signal.
		std::string detail = "helper died by signal " +
				     std::to_string(
					     outcome.checks.at(
						     "helper_signal"));
		if (!outcome.crash_rip.empty()) {
			detail += "; faulting RIP " + outcome.crash_rip +
				  " (syscall-server base " +
				  outcome.crash_base + ")";
		}
		for (const auto &[name, value] : outcome.checks) {
			if (name == "helper_signal")
				continue;
			detail += "; " + name + "=" + std::to_string(value);
		}
		FAIL(detail);
	}
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
