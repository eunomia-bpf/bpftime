#include "catch2/catch_test_macros.hpp"
#include <cstdint>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

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

// The helper caches the calling identity. A forked child inherits that cache,
// so without refreshing it in the child the helper keeps reporting the
// parent's pid and tid instead of the child's own.
TEST_CASE("bpf_get_current_pid_tgid reports the calling process after fork")
{
	REQUIRE(helper_tgid(call_helper()) ==
		static_cast<uint32_t>(getpid()));
	REQUIRE(helper_tid(call_helper()) ==
		static_cast<uint32_t>(syscall(SYS_gettid)));

	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		uint64_t value = call_helper();
		bool ok = helper_tgid(value) == static_cast<uint32_t>(getpid()) &&
			  helper_tid(value) ==
				  static_cast<uint32_t>(syscall(SYS_gettid));
		// A grandchild must also observe its own identity, not the
		// value inherited through two levels of fork.
		pid_t grandchild = fork();
		if (grandchild == 0) {
			uint64_t gvalue = call_helper();
			bool gok =
				helper_tgid(gvalue) ==
					static_cast<uint32_t>(getpid()) &&
				helper_tid(gvalue) ==
					static_cast<uint32_t>(syscall(SYS_gettid));
			_exit(gok ? 0 : 1);
		}
		int gstatus = 0;
		waitpid(grandchild, &gstatus, 0);
		_exit((ok && WIFEXITED(gstatus) &&
		       WEXITSTATUS(gstatus) == 0) ?
			      0 :
			      1);
	}

	int status = 0;
	waitpid(child, &status, 0);
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);

	// The parent's own cached identity must be unaffected by the children.
	REQUIRE(helper_tgid(call_helper()) ==
		static_cast<uint32_t>(getpid()));
	REQUIRE(helper_tid(call_helper()) ==
		static_cast<uint32_t>(syscall(SYS_gettid)));
}
