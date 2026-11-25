/* SPDX-License-Identifier: MIT */
#include "catch2/catch_test_macros.hpp"

#if defined(__linux__)
#include "bpftime_shm.hpp"
#include "syscall_context.hpp"
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <linux/bpf.h>
#include <linux/memfd.h>
#include <cerrno>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

namespace
{
struct shm_fixture {
	shm_fixture()
	{
		setenv("BPFTIME_GLOBAL_SHM_NAME",
		       ("bpftime-memfd-test-" +
			std::to_string(getpid()))
			       .c_str(),
		       1);
		bpftime_destroy_global_shm();
		bpftime_initialize_global_shm(
			bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);
	}
	~shm_fixture()
	{
		bpftime_destroy_global_shm();
		bpftime_initialize_global_shm(
			bpftime::shm_open_type::SHM_NO_CREATE);
		unsetenv("BPFTIME_GLOBAL_SHM_NAME");
	}
};
} // namespace

TEST_CASE("mocked memfd_create is backed by a writable kernel memfd",
	  "[syscall-server][memfd]")
{
	shm_fixture fixture;
	syscall_context ctx;

	int fd = ctx.handle_memfd_create("bpftime-memfd-unit-test", 0);
	REQUIRE(fd >= 0);

	// The descriptor must resolve to a real kernel memfd, not the
	// /dev/null fake fd used by the previous implementation.
	char link_path[64];
	char link_target[128] = {};
	snprintf(link_path, sizeof(link_path), "/proc/self/fd/%d", fd);
	ssize_t n = readlink(link_path, link_target,
			     sizeof(link_target) - 1);
	REQUIRE(n > 0);
	link_target[n] = '\0';
	// Kernels render the link as "memfd:<name>" or, on older backports,
	// "/memfd:<name> (deleted)"; either way it must not be /dev/null.
	REQUIRE(std::strstr(link_target, "memfd:") != nullptr);
	constexpr size_t region_size = 4096;
	// ftruncate fails with EINVAL on a /dev/null backing fd.
	REQUIRE(ftruncate(fd, region_size) == 0);

	const std::string payload = "bpftime memfd payload";
	REQUIRE(write(fd, payload.data(), payload.size()) ==
		(ssize_t)payload.size());
	REQUIRE(lseek(fd, 0, SEEK_SET) == 0);
	char read_back[64] = {};
	REQUIRE(read(fd, read_back, payload.size()) ==
		(ssize_t)payload.size());
	REQUIRE(std::string(read_back, payload.size()) == payload);

	// A writable MAP_SHARED mapping requires a regular, seekable file;
	// /dev/null rejects it with ENODEV.
	void *mapping = mmap(nullptr, region_size,
			     PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	REQUIRE(mapping != MAP_FAILED);
	auto *bytes = static_cast<unsigned char *>(mapping);
	REQUIRE(bytes[0] == static_cast<unsigned char>(payload[0]));
	bytes[1] = static_cast<unsigned char>(payload[1]) ^ 0x5a;
	REQUIRE(munmap(mapping, region_size) == 0);

	// #445 compatibility: dup3 of a BPF map fd onto the memfd slot
	// replaces the recorded handler while keeping the shared map usable.
	bpftime::bpf_map_attr attr{
		.type = BPF_MAP_TYPE_ARRAY, .key_size = sizeof(uint32_t),
		.value_size = sizeof(uint64_t), .max_ents = 1
	};
	int map_fd = bpftime_maps_create(-1, "memfd_dup_map", attr);
	REQUIRE(map_fd >= 0);
	REQUIRE_FALSE(bpftime_is_map_fd(fd));
	REQUIRE(ctx.handle_dup3(map_fd, fd, 0) == fd);
	REQUIRE(bpftime_is_map_fd(fd));
	uint32_t key = 0;
	uint64_t value = 0x1122334455667788ULL;
	REQUIRE(bpftime_map_update_elem(fd, &key, &value, 0) == 0);
	const void *observed = bpftime_map_lookup_elem(fd, &key);
	REQUIRE(observed != nullptr);
	REQUIRE(*static_cast<const uint64_t *>(observed) == value);

	REQUIRE(ctx.handle_close(map_fd) == 0);
	REQUIRE(ctx.handle_close(fd) == 0);
	// close returns the replaced slot to the unused state.
	REQUIRE_FALSE(bpftime_is_map_fd(fd));
}

TEST_CASE("mocked memfd_create relocates a standard descriptor above 2",
	  "[syscall-server][memfd]")
{
	shm_fixture fixture;
	// Run in a child so closing descriptor 0 cannot disturb the test
	// runner itself. The child inherits the initialized SHM mapping.
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		close(0);
		syscall_context ctx;
		int fd = ctx.handle_memfd_create("bpftime-memfd-relocate", 0);
		if (fd < 3)
			_exit(10);
		char link_path[64];
		char link_target[128] = {};
		snprintf(link_path, sizeof(link_path), "/proc/self/fd/%d",
			 fd);
		ssize_t n = readlink(link_path, link_target,
				     sizeof(link_target) - 1);
		if (n <= 0 || std::strstr(link_target, "memfd:") == nullptr)
			_exit(11);
		if (fcntl(fd, F_GETFD) < 0)
			_exit(12);
		ctx.handle_close(fd);
		_exit(0);
	}
	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
}

TEST_CASE("mocked memfd_create passes sealing flags to the kernel",
	  "[syscall-server][memfd]")
{
	shm_fixture fixture;
	syscall_context ctx;

	int sealed = ctx.handle_memfd_create("bpftime-memfd-sealed",
					     MFD_ALLOW_SEALING);
	REQUIRE(sealed >= 0);
	REQUIRE(ftruncate(sealed, 4096) == 0);
	REQUIRE(fcntl(sealed, F_ADD_SEALS, F_SEAL_SHRINK) == 0);
	int seals = fcntl(sealed, F_GET_SEALS);
	REQUIRE(seals >= 0);
	REQUIRE((seals & F_SEAL_SHRINK) != 0);

	// Without MFD_ALLOW_SEALING the kernel rejects sealing with EPERM,
	// proving the flag bits reach the real memfd unchanged.
	int plain = ctx.handle_memfd_create("bpftime-memfd-plain", 0);
	REQUIRE(plain >= 0);
	REQUIRE(ftruncate(plain, 4096) == 0);
	errno = 0;
	REQUIRE(fcntl(plain, F_ADD_SEALS, F_SEAL_SHRINK) == -1);
	REQUIRE(errno == EPERM);

	REQUIRE(ctx.handle_close(sealed) == 0);
	REQUIRE(ctx.handle_close(plain) == 0);
}

TEST_CASE("mocked memfd_create preserves MFD_CLOEXEC",
	  "[syscall-server][memfd]")
{
	shm_fixture fixture;
	syscall_context ctx;

	int fd = ctx.handle_memfd_create("bpftime-memfd-cloexec",
					 MFD_CLOEXEC);
	REQUIRE(fd >= 0);
	int flags = fcntl(fd, F_GETFD);
	REQUIRE(flags >= 0);
	REQUIRE((flags & FD_CLOEXEC) != 0);
	REQUIRE(ctx.handle_close(fd) == 0);
}
#endif
