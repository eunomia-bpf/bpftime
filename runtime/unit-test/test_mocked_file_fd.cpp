/* SPDX-License-Identifier: MIT
 *
 * Regression tests for mocked file descriptors.
 *
 * A mocked open(2)/openat(2) returns an unlinked kernel temp file. The
 * contract is that this fd behaves like an ordinary file for every reader:
 * libc read(2), a bare SYS_read (the Go runtime bypasses libc and issues
 * the syscall instruction directly), lseek(2), fstat(2) and close(2).
 *
 * The previous implementation kept the payload in a userspace map and left
 * the backing temp file empty, so a raw kernel read returned EOF and fstat
 * reported a zero size while a libc-interposed read returned content.
 */
#include "catch2/catch_test_macros.hpp"

#if defined(__linux__)
#include "bpftime_shm.hpp"
#include "syscall_context.hpp"
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace
{
struct shm_test_fixture {
	std::string shm_name;
	bool initialized = false;

	shm_test_fixture()
	{
		shm_name = "bpftime-mock-fd-test-" + std::to_string(getpid());
		REQUIRE(setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(),
				1) == 0);
		bpftime_destroy_global_shm();
		bpftime_initialize_global_shm(
			bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);
		initialized = true;
	}

	~shm_test_fixture()
	{
		if (initialized) {
			bpftime_destroy_global_shm();
			bpftime_initialize_global_shm(
				bpftime::shm_open_type::SHM_NO_CREATE);
		}
		unsetenv("BPFTIME_GLOBAL_SHM_NAME");
	}
};

// The uprobe type file is mocked unconditionally, so the expected payload is
// deterministic on every host.
constexpr const char *MOCKED_UPROBE_TYPE_PATH =
	"/sys/bus/event_source/devices/uprobe/type";
const std::string EXPECTED_UPROBE_TYPE =
	std::to_string(MOCKED_UPROBE_TYPE_VALUE);

long
bare_read(int fd, void *buf, size_t count)
{
	return syscall(SYS_read, fd, buf, count);
}
} // namespace

TEST_CASE("mocked openat fd is a populated, rewindable kernel file",
	  "[syscall-server][mock-file]")
{
	shm_test_fixture fixture;
	syscall_context ctx;

	int fd = ctx.handle_openat(AT_FDCWD, MOCKED_UPROBE_TYPE_PATH,
				   O_RDONLY, 0);
	REQUIRE(fd >= 0);

	// A bare SYS_read (Go-style, no libc interposition) must hit the
	// populated kernel file instead of an empty temp file.
	char raw[32] = {};
	REQUIRE(bare_read(fd, raw, sizeof(raw)) ==
		(long)EXPECTED_UPROBE_TYPE.size());
	REQUIRE(EXPECTED_UPROBE_TYPE == raw);

	// fstat reports the real payload size, not the old zero size.
	struct stat st = {};
	REQUIRE(fstat(fd, &st) == 0);
	REQUIRE((size_t)st.st_size == EXPECTED_UPROBE_TYPE.size());

	// One kernel cursor: rewinding makes the whole payload readable again.
	REQUIRE(lseek(fd, 0, SEEK_SET) == 0);
	char again[32] = {};
	REQUIRE(bare_read(fd, again, sizeof(again)) ==
		(long)EXPECTED_UPROBE_TYPE.size());
	REQUIRE(EXPECTED_UPROBE_TYPE == again);

	// Reading past the payload yields a normal kernel EOF.
	REQUIRE(bare_read(fd, again, sizeof(again)) == 0);

	REQUIRE(ctx.handle_close(fd) == 0);
}

TEST_CASE("libc read and raw SYS_read return the same mocked content",
	  "[syscall-server][mock-file]")
{
	shm_test_fixture fixture;
	syscall_context ctx;

	int fd = ctx.handle_open(
		MOCKED_UPROBE_TYPE_PATH, O_RDONLY | O_CLOEXEC, 0);
	REQUIRE(fd >= 0);

	char libc_buf[32] = {};
	REQUIRE(read(fd, libc_buf, sizeof(libc_buf)) ==
		(ssize_t)EXPECTED_UPROBE_TYPE.size());
	REQUIRE(EXPECTED_UPROBE_TYPE == libc_buf);

	REQUIRE(lseek(fd, 0, SEEK_SET) == 0);
	char raw_buf[32] = {};
	REQUIRE(bare_read(fd, raw_buf, sizeof(raw_buf)) ==
		(long)EXPECTED_UPROBE_TYPE.size());
	REQUIRE(EXPECTED_UPROBE_TYPE == raw_buf);

	REQUIRE(ctx.handle_close(fd) == 0);
}

TEST_CASE("unmocked files keep passing through to the real kernel fd",
	  "[syscall-server][mock-file]")
{
	shm_test_fixture fixture;
	syscall_context ctx;

	// /dev/null is not part of the mocked file set.
	int fd = ctx.handle_openat(AT_FDCWD, "/dev/null", O_RDONLY, 0);
	REQUIRE(fd >= 0);
	char buf[8] = {};
	REQUIRE(bare_read(fd, buf, sizeof(buf)) == 0);
	struct stat st = {};
	REQUIRE(fstat(fd, &st) == 0);
	REQUIRE(S_ISCHR(st.st_mode));
	REQUIRE(ctx.handle_close(fd) == 0);
}
#endif
