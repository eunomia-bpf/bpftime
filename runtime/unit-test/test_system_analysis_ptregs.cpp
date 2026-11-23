/* SPDX-License-Identifier: MIT
 *
 * Coverage for the zero-configuration sys_enter raw tracepoint emulation:
 *
 *  - in userspace mode BPF_RAW_TRACEPOINT_OPEN("sys_enter") is emulated by
 *    default, without requiring BPFTIME_SYSTEM_ANALYSIS_PTREGS_OFFSET;
 *  - an explicit positive offset overrides the default;
 *  - an explicit "0" and any non-sys_enter name keep passing the syscall
 *    through to the kernel instead of returning the emulated file fd;
 *  - a system_analysis BPF_MAP_LOOKUP_ELEM is completed with the effective
 *    pt_regs offset, including the built-in default when no env is set.
 */
#include "catch2/catch_test_macros.hpp"

#if defined(__linux__)
#include "bpftime_shm.hpp"
#include "syscall_context.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <linux/bpf.h>
#include <sys/stat.h>
#include <unistd.h>

namespace
{
// Mirrors the private layout in syscall_context.cpp.
struct system_analysis_compat_value {
	uint64_t address;
	uint32_t pid;
	int32_t error;
	uint8_t code[128];
};

constexpr unsigned long DEFAULT_PTREGS_OFFSET = 4096;
constexpr uint64_t STACK_BASE = 1ULL << 20;

struct shm_test_fixture {
	std::string shm_name;
	bool initialized = false;

	shm_test_fixture()
	{
		shm_name = "bpftime-ptregs-test-" + std::to_string(getpid());
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

void
clear_analysis_env()
{
	unsetenv("BPFTIME_SYSTEM_ANALYSIS_PTREGS_OFFSET");
	unsetenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_ADDRESS");
	unsetenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_OFFSET");
	unsetenv("BPFTIME_RUN_WITH_KERNEL");
}

int
open_raw_tracepoint(syscall_context &ctx, const char *name)
{
	union bpf_attr attr {};
	attr.raw_tracepoint.name = reinterpret_cast<uint64_t>(name);
	return ctx.handle_sysbpf(BPF_RAW_TRACEPOINT_OPEN, &attr, sizeof(attr));
}

// The emulated handle backs an unlinked regular file, while a kernel
// bpf fd is an anonymous inode and a rejected syscall returns -1.
void
require_passthrough_not_emulated(int fd)
{
	if (fd >= 0) {
		struct stat st {};
		REQUIRE(fstat(fd, &st) == 0);
		REQUIRE_FALSE(S_ISREG(st.st_mode));
		close(fd);
	} else {
		REQUIRE(errno == EPERM);
	}
}

int
create_analysis_map()
{
	bpftime::bpf_map_attr attr{};
	attr.type = BPF_MAP_TYPE_ARRAY;
	attr.key_size = sizeof(uint32_t);
	attr.value_size = sizeof(system_analysis_compat_value);
	attr.max_ents = 1;
	int fd = bpftime_maps_create(-1, "system_analysis", attr);
	REQUIRE(fd >= 0);
	return fd;
}

system_analysis_compat_value
lookup_analysis(syscall_context &ctx, int map_fd)
{
	system_analysis_compat_value result{};
	union bpf_attr attr {};
	uint32_t key = 0;
	attr.map_fd = static_cast<uint32_t>(map_fd);
	attr.key = reinterpret_cast<uint64_t>(&key);
	attr.value = reinterpret_cast<uint64_t>(&result);
	REQUIRE(ctx.handle_sysbpf(BPF_MAP_LOOKUP_ELEM, &attr, sizeof(attr)) ==
		0);
	return result;
}
} // namespace

TEST_CASE("sys_enter raw tracepoint is emulated by default in userspace mode",
	  "[syscall-server][system-analysis]")
{
	shm_test_fixture fixture;
	clear_analysis_env();
	syscall_context ctx;

	int fd = open_raw_tracepoint(ctx, "sys_enter");
	REQUIRE(fd >= 0);
	struct stat st {};
	REQUIRE(fstat(fd, &st) == 0);
	REQUIRE(S_ISREG(st.st_mode));
	close(fd);
}

TEST_CASE("explicit positive pt_regs offset stays honored",
	  "[syscall-server][system-analysis]")
{
	shm_test_fixture fixture;
	clear_analysis_env();
	REQUIRE(setenv("BPFTIME_SYSTEM_ANALYSIS_PTREGS_OFFSET", "8192", 1) ==
		0);
	syscall_context ctx;

	int fd = open_raw_tracepoint(ctx, "sys_enter");
	REQUIRE(fd >= 0);
	struct stat st {};
	REQUIRE(fstat(fd, &st) == 0);
	REQUIRE(S_ISREG(st.st_mode));
	close(fd);
}

TEST_CASE("pt_regs offset zero disables the raw tracepoint emulation",
	  "[syscall-server][system-analysis]")
{
	shm_test_fixture fixture;
	clear_analysis_env();
	REQUIRE(setenv("BPFTIME_SYSTEM_ANALYSIS_PTREGS_OFFSET", "0", 1) == 0);
	syscall_context ctx;

	errno = 0;
	require_passthrough_not_emulated(open_raw_tracepoint(ctx, "sys_enter"));
}

TEST_CASE("non-sys_enter raw tracepoints always pass through",
	  "[syscall-server][system-analysis]")
{
	shm_test_fixture fixture;
	clear_analysis_env();
	syscall_context ctx;

	errno = 0;
	require_passthrough_not_emulated(open_raw_tracepoint(ctx, "sys_exit"));
}

TEST_CASE("system_analysis lookup uses the default pt_regs offset without env",
	  "[syscall-server][system-analysis]")
{
	shm_test_fixture fixture;
	clear_analysis_env();
	syscall_context ctx;

	int map_fd = create_analysis_map();
	system_analysis_compat_value seed{};
	seed.pid = 4321;
	seed.address = 512;
	uint32_t key = 0;
	REQUIRE(bpftime_map_update_elem(map_fd, &key, &seed, 0) == 0);

	system_analysis_compat_value result = lookup_analysis(ctx, map_fd);
	REQUIRE(result.pid == 0);
	REQUIRE(result.error == 0);
	REQUIRE(result.address == STACK_BASE + DEFAULT_PTREGS_OFFSET);
	uint64_t encoded_stack_base = 0;
	memcpy(&encoded_stack_base, result.code, sizeof(encoded_stack_base));
	REQUIRE(encoded_stack_base == STACK_BASE);
}

TEST_CASE("system_analysis lookup honors an explicit pt_regs offset",
	  "[syscall-server][system-analysis]")
{
	shm_test_fixture fixture;
	clear_analysis_env();
	REQUIRE(setenv("BPFTIME_SYSTEM_ANALYSIS_PTREGS_OFFSET", "8192", 1) ==
		0);
	syscall_context ctx;

	int map_fd = create_analysis_map();
	system_analysis_compat_value seed{};
	seed.pid = 4321;
	seed.address = 512;
	uint32_t key = 0;
	REQUIRE(bpftime_map_update_elem(map_fd, &key, &seed, 0) == 0);

	system_analysis_compat_value result = lookup_analysis(ctx, map_fd);
	REQUIRE(result.pid == 0);
	REQUIRE(result.error == 0);
	REQUIRE(result.address == STACK_BASE + 8192);
}
#endif
