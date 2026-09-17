#include "catch2/catch_test_macros.hpp"
#include <cstdint>
#include <sys/wait.h>
#include <unistd.h>

#if __APPLE__
#include <pthread.h>
#else
#include <sys/syscall.h>
#endif

extern "C" {
uint64_t bpftime_get_current_pid_tgid(uint64_t, uint64_t, uint64_t, uint64_t,
				      uint64_t);
}

static uint64_t call_helper()
{
	return bpftime_get_current_pid_tgid(0, 0, 0, 0, 0);
}

static uint32_t helper_tgid(uint64_t v)
{
	return static_cast<uint32_t>(v >> 32);
}

static uint32_t helper_tid(uint64_t v)
{
	return static_cast<uint32_t>(v);
}

// Mirror the identity the helper is expected to report on this platform.
static uint32_t current_tid()
{
#if __APPLE__
	uint64_t tid = 0;
	pthread_threadid_np(NULL, &tid);
	return static_cast<uint32_t>(tid);
#else
	return static_cast<uint32_t>(syscall(SYS_gettid));
#endif
}

static bool helper_matches_this_process()
{
	uint64_t value = call_helper();
	return helper_tgid(value) == static_cast<uint32_t>(getpid()) &&
	       helper_tid(value) == current_tid();
}

// The helper caches the calling identity. A forked child inherits that cache,
// so without refreshing it in the child the helper keeps reporting the
// parent's pid and tid instead of the child's own.
TEST_CASE("bpf_get_current_pid_tgid reports the calling process after fork")
{
	REQUIRE(helper_matches_this_process());

	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		bool ok = helper_matches_this_process();
		// A grandchild must also observe its own identity, not the
		// value inherited through two levels of fork.
		pid_t grandchild = fork();
		if (grandchild < 0) {
			_exit(1);
		}
		if (grandchild == 0) {
			_exit(helper_matches_this_process() ? 0 : 1);
		}
		int gstatus = 0;
		bool grandchild_ok = waitpid(grandchild, &gstatus, 0) ==
					     grandchild &&
				     WIFEXITED(gstatus) &&
				     WEXITSTATUS(gstatus) == 0;
		_exit((ok && grandchild_ok) ? 0 : 1);
	}

	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);

	// The parent's own cached identity must be unaffected by the children.
	REQUIRE(helper_matches_this_process());
}
