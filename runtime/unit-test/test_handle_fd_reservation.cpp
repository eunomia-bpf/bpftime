/* SPDX-License-Identifier: MIT
 *
 * Regression tests for mocked BPF handle fd reservation.
 *
 * BPF_BTF_LOAD, BPF_MAP_CREATE, BPF_PROG_LOAD and BPF_LINK_CREATE only
 * need a unique kernel fd number as a handle slot. The reservation must:
 *   - be backed by /dev/null instead of a created-and-unlinked temp file,
 *   - carry FD_CLOEXEC so handles never leak across execve,
 *   - return an fd >= 3 even when a standard descriptor is closed,
 *   - never leave a /tmp/bpftime-mock.* dentry behind.
 * BPF_RAW_TRACEPOINT_OPEN keeps a regular-file backing on purpose; see the
 * fstat() contract in test_system_analysis_ptregs.cpp.
 */
#include "catch2/catch_test_macros.hpp"
#if defined(__linux__)
#include "bpftime_shm.hpp"
#include "syscall_context.hpp"
#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <filesystem>
#include <linux/bpf.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace
{
struct reservation_shm_fixture {
	std::string shm_name;
	bool initialized = false;

	reservation_shm_fixture()
	{
		// Honor an externally provided shm name (parallel test
		// lanes isolate themselves this way); otherwise derive a
		// process-unique one.
		const char *existing = getenv("BPFTIME_GLOBAL_SHM_NAME");
		shm_name = existing != nullptr ?
			std::string(existing) :
			("bpftime_test_r3_" + std::to_string(getpid()));
		REQUIRE(setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(),
				1) == 0);
		bpftime_destroy_global_shm();
		bpftime_initialize_global_shm(
			bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);
		initialized = true;
	}

	~reservation_shm_fixture()
	{
		if (initialized) {
			bpftime_destroy_global_shm();
			bpftime_initialize_global_shm(
				bpftime::shm_open_type::SHM_NO_CREATE);
		}
	}
};

std::vector<std::string>
snapshot_mock_tmp_entries()
{
	std::vector<std::string> entries;
	std::error_code ec;
	for (const auto &path :
	     std::filesystem::directory_iterator("/tmp", ec)) {
		const std::string name = path.path().filename().string();
		if (name.rfind("bpftime-mock.", 0) == 0)
			entries.push_back(name);
	}
	return entries;
}

bool fd_is_cloexec(int fd)
{
	int flags = fcntl(fd, F_GETFD);
	return flags >= 0 && (flags & FD_CLOEXEC) != 0;
}

std::string fd_backing_path(int fd)
{
	const std::string link =
		"/proc/self/fd/" + std::to_string(fd);
	char buf[128] = {};
	ssize_t n = readlink(link.c_str(), buf, sizeof(buf) - 1);
	if (n < 0)
		return {};
	return std::string(buf, static_cast<size_t>(n));
}

// True while any open descriptor of this process still points at a
// bpftime-mock temp file (including unlinked ones, which readlink
// reports with a " (deleted)" suffix). Unlike a /tmp dentry snapshot
// this is race-free against foreign processes and fails on the old
// mkostemp-based reservation while the handle is open.
bool any_open_mock_tmp_fd()
{
	std::error_code ec;
	for (const auto &entry :
	     std::filesystem::directory_iterator("/proc/self/fd", ec)) {
		char buf[128] = {};
	ssize_t n = readlink(entry.path().c_str(), buf,
				     sizeof(buf) - 1);
		if (n > 0 &&
		    std::string(buf, static_cast<size_t>(n))
			    .find("bpftime-mock.") !=
			    std::string::npos)
			return true;
	}
	return false;
}

void require_reserved_handle(int fd)
{
	REQUIRE(fd >= 3);
	REQUIRE(fd_is_cloexec(fd));
	REQUIRE(fd_backing_path(fd) == "/dev/null");
	REQUIRE_FALSE(any_open_mock_tmp_fd());
}
} // namespace

TEST_CASE("mocked bpf handles reserve cloexec /dev/null fds without temp files",
	  "[syscall-server][handle-fd]")
{
	reservation_shm_fixture fixture;
	syscall_context ctx;

	const auto before = snapshot_mock_tmp_entries();

	SECTION("BPF_BTF_LOAD")
	{
		union bpf_attr attr {};
		int fd = (int)ctx.handle_sysbpf(BPF_BTF_LOAD, &attr,
						 sizeof(attr));
		require_reserved_handle(fd);
		REQUIRE(ctx.handle_close(fd) == 0);
	}

	SECTION("BPF_MAP_CREATE and the shm open_fake_fd path")
	{
		union bpf_attr attr {};
		attr.map_type = BPF_MAP_TYPE_ARRAY;
		attr.key_size = sizeof(uint32_t);
		attr.value_size = sizeof(uint64_t);
		attr.max_entries = 1;
		int map_fd = (int)ctx.handle_sysbpf(BPF_MAP_CREATE, &attr,
						     sizeof(attr));
		require_reserved_handle(map_fd);
		REQUIRE(ctx.handle_close(map_fd) == 0);

		// bpftime_maps_create with fd == -1 allocates through
		// bpftime_shm::open_fake_fd directly.
		bpftime::bpf_map_attr shm_attr {
			.type = BPF_MAP_TYPE_ARRAY,
			.key_size = sizeof(uint32_t),
			.value_size = sizeof(uint64_t),
			.max_ents = 1
		};
		int shm_fd = bpftime_maps_create(-1, "r3_reservation",
						  shm_attr);
		require_reserved_handle(shm_fd);
		REQUIRE(ctx.handle_close(shm_fd) == 0);
	}

	SECTION("BPF_PROG_LOAD")
	{
		struct bpf_insn insns[2] = {
			{ .code = (uint8_t)(BPF_ALU64 | BPF_MOV | BPF_K),
			  .dst_reg = BPF_REG_0,
			  .src_reg = 0,
			  .off = 0,
			  .imm = 0 },
			{ .code = (uint8_t)(BPF_JMP | BPF_EXIT),
			  .dst_reg = 0,
			  .src_reg = 0,
			  .off = 0,
			  .imm = 0 },
		};
		char license[] = "MIT";
		union bpf_attr attr {};
		attr.prog_type = BPF_PROG_TYPE_KPROBE;
		attr.insn_cnt = 2;
		attr.insns = (uint64_t)(uintptr_t)insns;
		attr.license = (uint64_t)(uintptr_t)license;
		int prog_fd = (int)ctx.handle_sysbpf(BPF_PROG_LOAD, &attr,
						      sizeof(attr));
		require_reserved_handle(prog_fd);
		REQUIRE(ctx.handle_close(prog_fd) == 0);
	}

	SECTION("BPF_LINK_CREATE reservation happens before validation")
	{
		// The link itself cannot be created without valid prog and
		// target fds, but the handle reservation runs first, so the
		// call must still leave no temp-file dentry behind.
		union bpf_attr attr {};
		attr.link_create.prog_fd = (uint32_t)-1;
		attr.link_create.target_fd = (uint32_t)-1;
		attr.link_create.attach_type = BPF_PERF_EVENT;
		// Validation rejects the invalid fds, so the call fails and
		// the failure path releases the reserved fd without leaving
		// a bpftime-mock descriptor.
		REQUIRE(ctx.handle_sysbpf(BPF_LINK_CREATE, &attr,
					   sizeof(attr)) == -1);
	}

	// BPF_RAW_TRACEPOINT_OPEN is intentionally not covered here and not
	// converted to /dev/null: its emulated handle must back an unlinked
	// regular file so fstat() distinguishes it from a kernel anonymous
	// inode or a rejected syscall. That contract is asserted in
	// test_system_analysis_ptregs.cpp, which keeps using
	// create_unlinked_mock_fd() for this one path.

	const auto after = snapshot_mock_tmp_entries();
	for (const auto &name : after) {
		// No new bpftime-mock dentry may appear; entries created by
		// unrelated processes before this test ran are allowed.
		REQUIRE(std::find(before.begin(), before.end(), name) !=
			before.end());
	}
}

TEST_CASE("handle reservation moves low fds above fd 2",
	  "[syscall-server][handle-fd]")
{
	reservation_shm_fixture fixture;

	SECTION("integration: handles stay >= 3 with stdin closed")
	{
		pid_t pid = fork();
		REQUIRE(pid >= 0);
		if (pid == 0) {
			// Child: no Catch macros; report through exit status.
			// Framework startup may legitimately reuse a closed
			// std fd, so only assert the handle contract itself.
			close(STDIN_FILENO);
			syscall_context ctx;

			union bpf_attr attr {};
			int btf_fd = (int)ctx.handle_sysbpf(
				BPF_BTF_LOAD, &attr, sizeof(attr));
			bool btf_ok = btf_fd >= 3 && fd_is_cloexec(btf_fd) &&
				      fd_backing_path(btf_fd) ==
					      "/dev/null";
			if (btf_fd >= 0)
				(void)ctx.handle_close(btf_fd);

			bpftime::bpf_map_attr shm_attr {
				.type = BPF_MAP_TYPE_ARRAY,
				.key_size = sizeof(uint32_t),
				.value_size = sizeof(uint64_t),
				.max_ents = 1
			};
			int map_fd = bpftime_maps_create(
				-1, "r3_stdin_closed", shm_attr);
			bool map_ok = map_fd >= 3 && fd_is_cloexec(map_fd) &&
				      fd_backing_path(map_fd) ==
					      "/dev/null";
			if (map_fd >= 0)
				(void)ctx.handle_close(map_fd);

			_exit(btf_ok && map_ok ? 0 : 1);
		}

		int status = 0;
		REQUIRE(waitpid(pid, &status, 0) == pid);
		REQUIRE(WIFEXITED(status));
		REQUIRE(WEXITSTATUS(status) == 0);
	}

	SECTION("raw move branch: open_fake_fd releases the low fd")
	{
		// The global shm was initialized by the fixture and is
		// inherited across fork(). Without constructing a
		// syscall_context (whose startup reuses low fds itself),
		// close stdin/stdout so the next open() must return a
		// descriptor <= 2, exercising the F_DUPFD_CLOEXEC move in
		// bpftime_shm::open_fake_fd(). Keep stderr for diagnostics.
		pid_t pid = fork();
		REQUIRE(pid >= 0);
		if (pid == 0) {
			close(STDIN_FILENO);
			close(STDOUT_FILENO);

			bpftime::bpf_map_attr shm_attr {
				.type = BPF_MAP_TYPE_ARRAY,
				.key_size = sizeof(uint32_t),
				.value_size = sizeof(uint64_t),
				.max_ents = 1
			};
			int fd = bpftime_maps_create(-1, "r3_low_fd_move",
						     shm_attr);
			bool ok = fd >= 3 && fd_is_cloexec(fd) &&
				  fd_backing_path(fd) == "/dev/null" &&
				  fcntl(STDIN_FILENO, F_GETFD) == -1 &&
				  errno == EBADF &&
				  fcntl(STDOUT_FILENO, F_GETFD) == -1;
			if (fd >= 0)
				close(fd);
			_exit(ok ? 0 : 1);
		}

		int status = 0;
		REQUIRE(waitpid(pid, &status, 0) == pid);
		REQUIRE(WIFEXITED(status));
		REQUIRE(WEXITSTATUS(status) == 0);
	}
}
#endif
