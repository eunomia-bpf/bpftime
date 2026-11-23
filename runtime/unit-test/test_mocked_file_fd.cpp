/* SPDX-License-Identifier: MIT */
#include "catch2/catch_test_macros.hpp"

#if defined(__linux__)
#include "bpftime_shm.hpp"
#if defined(__x86_64__) || defined(__aarch64__)
#include "synthetic_kallsyms.hpp"
#endif
#include "syscall_context.hpp"
#include <cstring>
#include <fcntl.h>
#include <linux/bpf.h>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>

namespace
{
struct shm_fixture {
	shm_fixture()
	{
		setenv("BPFTIME_GLOBAL_SHM_NAME",
		       ("bpftime-mock-fd-test-" +
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
#if defined(__x86_64__) || defined(__aarch64__)
struct analysis_value {
	uint64_t address;
	uint32_t pid;
	int32_t error;
	uint8_t code[128];
};
#endif
} // namespace

TEST_CASE("populated mock file fd and synthetic tpbase completion",
	  "[syscall-server][mock-file][system-analysis]")
{
	shm_fixture fixture;
	syscall_context ctx;
	const std::string expected =
		std::to_string(MOCKED_UPROBE_TYPE_VALUE);
	int fd = ctx.handle_openat(
		AT_FDCWD, "/sys/bus/event_source/devices/uprobe/type",
		O_RDONLY, 0);
	REQUIRE(fd >= 0);
	// A bare SYS_read (no libc interposition) hits the populated kernel file.
	char raw[32] = {};
	REQUIRE(syscall(SYS_read, fd, raw, sizeof(raw)) ==
		(long)expected.size());
	REQUIRE(expected == raw);
	// One kernel cursor: rewind makes the payload readable again.
	REQUIRE(lseek(fd, 0, SEEK_SET) == 0);
	REQUIRE(syscall(SYS_read, fd, raw, sizeof(raw)) ==
		(long)expected.size());
	REQUIRE(ctx.handle_close(fd) == 0);
#if defined(__x86_64__) || defined(__aarch64__)
	ctx.try_startup();
	unsetenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_ADDRESS");
	unsetenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_OFFSET");
	unsetenv("BPFTIME_SYSTEM_ANALYSIS_PTREGS_OFFSET");
	unsetenv("BPFTIME_RUN_WITH_KERNEL");
	bpftime::bpf_map_attr attr{
		.type = BPF_MAP_TYPE_ARRAY, .key_size = sizeof(uint32_t),
		.value_size = sizeof(analysis_value), .max_ents = 1
	};
	int map_fd = bpftime_maps_create(-1, "system_analysis", attr);
	REQUIRE(map_fd >= 0);
#if defined(__x86_64__)
	constexpr uint64_t addr =
		bpftime::SYNTHETIC_X86_FSBASE_WRITE_TASK_ADDR;
#elif defined(__aarch64__)
	constexpr uint64_t addr = bpftime::SYNTHETIC_ARM64_TLS_SET_ADDR;
#endif
	analysis_value value{ .address = addr, .pid = 4321 };
	uint32_t key = 0;
	REQUIRE(bpftime_map_update_elem(map_fd, &key, &value, 0) == 0);
	analysis_value result{};
	union bpf_attr lookup{
		.map_fd = (uint32_t)map_fd, .key = (uint64_t)&key,
		.value = (uint64_t)&result
	};
	REQUIRE(ctx.handle_sysbpf(BPF_MAP_LOOKUP_ELEM, &lookup,
				  sizeof(lookup)) == 0);
	REQUIRE(result.pid == 0);
	REQUIRE(result.address == 0);
	uint32_t off = 0;
#if defined(__x86_64__)
	REQUIRE(result.code[0] == 0x48);
	REQUIRE(result.code[1] == 0x89);
	REQUIRE(result.code[2] == 0xb7);
	memcpy(&off, &result.code[3], sizeof(off));
#elif defined(__aarch64__)
	uint32_t insn = 0;
	memcpy(&insn, result.code, sizeof(insn));
	REQUIRE(insn == (0xf9400000U |
			 (((bpftime_get_current_task_tpbase_offset() / 8U) &
			   0xfffU)
			  << 10)));
	off = (uint32_t)bpftime_get_current_task_tpbase_offset();
#endif
	REQUIRE(off == (uint32_t)bpftime_get_current_task_tpbase_offset());
#endif
}
#endif
