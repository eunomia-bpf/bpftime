/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#include "catch2/catch_test_macros.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <limits.h>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" uint64_t bpftime_get_current_comm(uint64_t buf,
					     uint64_t size, uint64_t,
					     uint64_t, uint64_t);

extern "C" void bpftime_copy_comm(char *buf, size_t size,
				        const char *comm);

namespace
{
#if defined(__linux__)
// A basename longer than the small-string-optimization capacity of the
// common libstdc++ std::string (15 bytes). With the previous cached
// std::string implementation the first helper call had to allocate on the
// heap for this name.
constexpr const char *long_executable_name =
	"bpftime-runtime-comm-reentry-probe-0123456789abcdef";
#endif

std::string current_exe_basename()
{
	char path[PATH_MAX];
	auto len = readlink("/proc/self/exe", path, sizeof(path) - 1);
	REQUIRE(len > 0);
	path[len] = 0;
	std::string full(path);
	auto slash = full.find_last_of('/');
	return full.substr(slash + 1);
}
} // namespace

TEST_CASE("bpf_get_current_comm returns the executable basename",
	  "[helpers][comm]")
{
	char comm[256];
	std::memset(comm, 0x5a, sizeof(comm));
	REQUIRE(bpftime_get_current_comm(reinterpret_cast<uint64_t>(comm),
				     sizeof(comm), 0, 0, 0) == 0);
	REQUIRE(std::strlen(comm) < sizeof(comm));
	REQUIRE(std::string(comm) == current_exe_basename());

	// Repeated calls stay consistent without any process-global cache.
	char again[256];
	REQUIRE(bpftime_get_current_comm(reinterpret_cast<uint64_t>(again),
				     sizeof(again), 0, 0, 0) == 0);
	REQUIRE(std::string(again) == std::string(comm));
}

TEST_CASE("bpf_get_current_comm always NUL-terminates the buffer",
	  "[helpers][comm]")
{
	const auto expected = current_exe_basename();
	char comm[8];
	std::memset(comm, 0x5a, sizeof(comm));
	REQUIRE(bpftime_get_current_comm(reinterpret_cast<uint64_t>(comm),
				     sizeof(comm), 0, 0, 0) == 0);
	// Kernel semantics: the destination is always NUL-terminated,
	// truncating the name to size - 1 bytes when it does not fit.
	REQUIRE(comm[sizeof(comm) - 1] == '\0');
	const size_t prefix_len =
		std::min(sizeof(comm) - 1, expected.size());
	REQUIRE(std::memcmp(comm, expected.data(), prefix_len) == 0);
	REQUIRE(std::strlen(comm) == prefix_len);
}

TEST_CASE("bpftime_copy_comm matches kernel comm-copy semantics",
	  "[helpers][comm]")
{
	// Short name: copied in full and terminated; strncpy zero-pads the
	// remaining tail.
	{
		char buf[16];
		std::memset(buf, 0x5a, sizeof(buf));
		bpftime_copy_comm(buf, sizeof(buf), "short");
		REQUIRE(std::string(buf) == "short");
		REQUIRE(buf[5] == '\0');
		REQUIRE(buf[6] == '\0');
	}

	// Long name: truncated to size - 1 bytes and force-terminated.
	{
		const char long_name[] =
			"bpftime-runtime-comm-reentry-probe-0123456789abcdef";
		char buf[8];
		std::memset(buf, 0x5a, sizeof(buf));
		bpftime_copy_comm(buf, sizeof(buf), long_name);
		REQUIRE(std::memcmp(buf, long_name, sizeof(buf) - 1) == 0);
		REQUIRE(buf[sizeof(buf) - 1] == '\0');
		REQUIRE(std::strlen(buf) == sizeof(buf) - 1);
	}

	// size == 1: only the terminator fits.
	{
		char buf[1] = { static_cast<char>(0x5a) };
		bpftime_copy_comm(buf, sizeof(buf), "name");
		REQUIRE(buf[0] == '\0');
	}

	// size == 0: the destination must not be written.
	{
		char buf[4];
		char sentinel[4];
		std::memset(buf, 0x5a, sizeof(buf));
		std::memset(sentinel, 0x5a, sizeof(sentinel));
		bpftime_copy_comm(buf, 0, "name");
		REQUIRE(std::memcmp(buf, sentinel, sizeof(buf)) == 0);
	}
}

#if defined(__linux__)
// The reentry check runs in a child process whose executable path uses the
// long basename above. /proc/self/exe resolves symlinks to the target, so a
// hard link is used (with a real copy fallback across filesystems) instead
// of a symlink. The --wrap=malloc/calloc/realloc link options route every
// allocator entry through the counters below, so the assertion proves that
// the helper performs no heap allocation while the tracking window is open.
namespace
{
thread_local bool comm_alloc_tracking = false;
thread_local unsigned comm_alloc_count = 0;

constexpr const char *reentry_env = "BPFTIME_TEST_COMM_REENTRY";
} // namespace

extern "C" {
void *__real_malloc(size_t size) noexcept;
void *__real_calloc(size_t nmemb, size_t size) noexcept;
void *__real_realloc(void *ptr, size_t size) noexcept;
// Declared by glibc in <unistd.h> only under feature-test macros; state it
// directly so the child can hand the current environment to execve.
extern char **environ;

void *__wrap_malloc(size_t size) noexcept
{
	if (comm_alloc_tracking)
		++comm_alloc_count;
	return __real_malloc(size);
}

void *__wrap_calloc(size_t nmemb, size_t size) noexcept
{
	if (comm_alloc_tracking)
		++comm_alloc_count;
	return __real_calloc(nmemb, size);
}

void *__wrap_realloc(void *ptr, size_t size) noexcept
{
	if (comm_alloc_tracking)
		++comm_alloc_count;
	return __real_realloc(ptr, size);
}
}

static int run_comm_reentry_self_check()
{
	char comm[256];
	comm_alloc_count = 0;
	comm_alloc_tracking = true;
	auto rc = bpftime_get_current_comm(reinterpret_cast<uint64_t>(comm),
				       sizeof(comm), 0, 0, 0);
	comm_alloc_tracking = false;
	if (rc != 0)
		return 3;
	if (comm_alloc_count != 0)
		return 2;
	if (std::string(comm) != long_executable_name)
		return 4;
	return 0;
}

TEST_CASE("bpf_get_current_comm makes no heap allocation on first call",
	  "[helpers][comm][reentry]")
{
	if (getenv(reentry_env) != nullptr)
		std::exit(run_comm_reentry_self_check());

	char self[PATH_MAX];
	auto len = readlink("/proc/self/exe", self, sizeof(self) - 1);
	REQUIRE(len > 0);
	self[len] = 0;

	char dir_template[] = "/tmp/bpftime-comm-XXXXXX";
	char *dir = mkdtemp(dir_template);
	REQUIRE(dir != nullptr);
	const std::string link_path =
		std::string(dir) + "/" + long_executable_name;

	// A symlink would be resolved by /proc/self/exe back to this binary.
	// Prefer a hard link; fall back to a byte copy on cross-device or
	// link-restricted filesystems.
	if (::link(self, link_path.c_str()) != 0) {
		REQUIRE((errno == EXDEV || errno == EPERM));
		std::ifstream in(self, std::ios::binary);
		std::ofstream out(link_path, std::ios::binary | std::ios::trunc);
		REQUIRE(in.is_open());
		REQUIRE(out.is_open());
		out << in.rdbuf();
		out.close();
		REQUIRE(static_cast<bool>(out));
		REQUIRE(::chmod(link_path.c_str(), 0755) == 0);
	}

	pid_t pid = fork();
	REQUIRE(pid >= 0);
	if (pid == 0) {
		setenv(reentry_env, "1", 1);
		// Run only this test case in the child: without a test spec it
		// would run the whole suite first and fail on any test the host
		// cannot run before reaching the self check.
		char test_name[] =
			"bpf_get_current_comm makes no heap allocation on first call";
		char *const argv[] = { const_cast<char *>(link_path.c_str()),
				       test_name, nullptr };
		execve(link_path.c_str(), argv, environ);
		_exit(5);
	}

	int status = 0;
	REQUIRE(waitpid(pid, &status, 0) == pid);
	::unlink(link_path.c_str());
	::rmdir(dir);
	REQUIRE(WIFEXITED(status));
	// Exit code 2 means the helper entered the allocator; 3/4/5 mark
	// fixture or behavior failures described in run_comm_reentry_self_check.
	REQUIRE(WEXITSTATUS(status) == 0);
}
#endif
