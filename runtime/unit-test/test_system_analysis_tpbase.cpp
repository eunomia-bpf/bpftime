/* SPDX-License-Identifier: MIT
 *
 * Coverage for the zero-configuration thread-pointer (tpbase) fallback:
 *
 *  - the synthetic /proc/kallsyms table exposes the stable symbol addresses
 *    a binary-analysis tracer looks up when kernel BTF is unusable;
 *  - the arm64 tls_set instruction encoder follows the LDR immediate
 *    contract decoded by the userspace analyzer;
 *  - a system_analysis BPF_MAP_LOOKUP_ELEM at a synthetic symbol address is
 *    completed with the synthesized instruction and the BTF/fallback
 *    tpbase offset, while explicit env overrides keep working and kernel
 *    mode passes through untouched.
 */
#include "catch2/catch_test_macros.hpp"

#if defined(__linux__)
#include "bpftime_shm.hpp"
#include "current_task_tpbase.hpp"
#include "synthetic_kallsyms.hpp"
#include "syscall_context.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <linux/bpf.h>
#include <string>
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

struct shm_test_fixture {
	std::string shm_name;
	bool initialized = false;

	shm_test_fixture()
	{
		shm_name = "bpftime-tpbase-test-" + std::to_string(getpid());
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

int
create_analysis_map(const char *name)
{
	bpftime::bpf_map_attr attr{};
	attr.type = BPF_MAP_TYPE_ARRAY;
	attr.key_size = sizeof(uint32_t);
	attr.value_size = sizeof(system_analysis_compat_value);
	attr.max_ents = 1;
	int fd = bpftime_maps_create(-1, name, attr);
	REQUIRE(fd >= 0);
	return fd;
}

void
seed_analysis_value(int map_fd, uint32_t pid, uint64_t address)
{
	system_analysis_compat_value value{};
	value.pid = pid;
	value.address = address;
	uint32_t key = 0;
	REQUIRE(bpftime_map_update_elem(map_fd, &key, &value, 0) == 0);
}

system_analysis_compat_value
lookup_analysis(syscall_context &ctx, int map_fd)
{
	union bpf_attr attr{};
	uint32_t key = 0;
	system_analysis_compat_value result{};
	attr.map_fd = static_cast<uint32_t>(map_fd);
	attr.key = reinterpret_cast<uint64_t>(&key);
	attr.value = reinterpret_cast<uint64_t>(&result);
	REQUIRE(ctx.handle_sysbpf(BPF_MAP_LOOKUP_ELEM, &attr, sizeof(attr)) ==
		0);
	return result;
}

std::string
to_hex_address(uint64_t value)
{
	const char *digits = "0123456789abcdef";
	std::string out(16, '0');
	for (int i = 15; i >= 0; i--) {
		out[i] = digits[value & 0xf];
		value >>= 4;
	}
	return out;
}

bool
table_has_symbol(const std::string &table, const char *symbol,
		 uint64_t expected_address)
{
	const std::string entry =
		to_hex_address(expected_address) + " T " + symbol + "\n";
	return table.find(entry) != std::string::npos;
}
} // namespace

TEST_CASE("synthetic kallsyms exposes stable binary-analysis symbols",
	  "[syscall-server][system-analysis]")
{
	const std::string &table = bpftime::synthetic_kallsyms_content();
	REQUIRE(table_has_symbol(
			table, "x86_fsbase_write_task",
			bpftime::SYNTHETIC_X86_FSBASE_WRITE_TASK_ADDR));
	REQUIRE(table_has_symbol(table, "tls_set",
				 bpftime::SYNTHETIC_ARM64_TLS_SET_ADDR));
}

TEST_CASE("arm64 tls_set encoder emits the LDR immediate contract",
	  "[syscall-server][system-analysis]")
{
	// ldr x0, [x0, #2048]: imm12 = 2048/8 = 256, Rt=Rn=0, 64-bit LDR.
	REQUIRE(bpftime::encode_arm64_tls_set_insn(2048) == 0xf9440000U);
	// ldr x0, [x0, #2984]: imm12 = 373.
	REQUIRE(bpftime::encode_arm64_tls_set_insn(2984) ==
		(0xf9400000U | (373U << 10)));
	// The decoder rejects offsets below 64 or above 64 KiB; the fallback
	// and every BTF-derived slot stay inside that window.
	REQUIRE(bpftime::FALLBACK_CURRENT_TASK_TPBASE_OFFSET >= 64);
	REQUIRE(bpftime::FALLBACK_CURRENT_TASK_TPBASE_OFFSET < 65536);
	REQUIRE(bpftime::FALLBACK_CURRENT_TASK_TPBASE_OFFSET % 8U == 0);
}

TEST_CASE("tpbase resolver returns a decoder-acceptable offset",
	  "[syscall-server][system-analysis]")
{
	const uint32_t offset = bpftime::resolve_current_task_tpbase_offset();
#if defined(__x86_64__)
	REQUIRE(offset >= 500);
	REQUIRE(offset <= 20000);
#elif defined(__aarch64__)
	REQUIRE(offset >= 64);
	REQUIRE(offset < 65536);
	REQUIRE(offset % 8U == 0);
#endif
}

TEST_CASE("synthetic system_analysis request completes without any env switch",
	  "[syscall-server][system-analysis]")
{
	shm_test_fixture fixture;
	syscall_context ctx;
	unsetenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_ADDRESS");
	unsetenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_OFFSET");
	unsetenv("BPFTIME_SYSTEM_ANALYSIS_PTREGS_OFFSET");
	unsetenv("BPFTIME_RUN_WITH_KERNEL");

	const int map_fd = create_analysis_map("system_analysis");
#if defined(__x86_64__)
	seed_analysis_value(
		map_fd, 4321,
		bpftime::SYNTHETIC_X86_FSBASE_WRITE_TASK_ADDR);
	system_analysis_compat_value result = lookup_analysis(ctx, map_fd);
	REQUIRE(result.pid == 0);
	REQUIRE(result.error == 0);
	REQUIRE(result.address == 0);
	REQUIRE(result.code[0] == 0x48);
	REQUIRE(result.code[1] == 0x89);
	REQUIRE(result.code[2] == 0xb7);
	uint32_t encoded = 0;
	memcpy(&encoded, &result.code[3], sizeof(encoded));
	REQUIRE(encoded == bpftime::resolve_current_task_tpbase_offset());
#elif defined(__aarch64__)
	seed_analysis_value(map_fd, 4321,
			    bpftime::SYNTHETIC_ARM64_TLS_SET_ADDR);
	system_analysis_compat_value result = lookup_analysis(ctx, map_fd);
	REQUIRE(result.pid == 0);
	REQUIRE(result.error == 0);
	REQUIRE(result.address == 0);
	uint32_t insn = 0;
	memcpy(&insn, result.code, sizeof(insn));
	REQUIRE(insn == bpftime::encode_arm64_tls_set_insn(
				bpftime::resolve_current_task_tpbase_offset()));
#endif
}

#if defined(__x86_64__)
TEST_CASE("explicit tpbase env override still takes effect",
	  "[syscall-server][system-analysis]")
{
	shm_test_fixture fixture;
	syscall_context ctx;
	constexpr uint64_t custom_address = 0xffffffff8120abcdULL;
	constexpr uint32_t custom_offset = 2176;
	REQUIRE(setenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_ADDRESS",
			"0xffffffff8120abcd", 1) == 0);
	REQUIRE(setenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_OFFSET", "2176", 1) ==
		0);
	unsetenv("BPFTIME_SYSTEM_ANALYSIS_PTREGS_OFFSET");
	unsetenv("BPFTIME_RUN_WITH_KERNEL");

	const int map_fd = create_analysis_map("system_analysis");
	seed_analysis_value(map_fd, 4321, custom_address);
	system_analysis_compat_value result = lookup_analysis(ctx, map_fd);
	REQUIRE(result.pid == 0);
	REQUIRE(result.code[0] == 0x48);
	REQUIRE(result.code[1] == 0x89);
	REQUIRE(result.code[2] == 0xb7);
	uint32_t encoded = 0;
	memcpy(&encoded, &result.code[3], sizeof(encoded));
	REQUIRE(encoded == custom_offset);
	unsetenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_ADDRESS");
	unsetenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_OFFSET");
}
#endif

TEST_CASE("unrelated high-address analysis requests stay untouched",
	  "[syscall-server][system-analysis]")
{
	shm_test_fixture fixture;
	syscall_context ctx;
	unsetenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_ADDRESS");
	unsetenv("BPFTIME_SYSTEM_ANALYSIS_TPBASE_OFFSET");
	unsetenv("BPFTIME_SYSTEM_ANALYSIS_PTREGS_OFFSET");
	unsetenv("BPFTIME_RUN_WITH_KERNEL");

	const int map_fd = create_analysis_map("system_analysis");
	constexpr uint64_t unknown_address = 0xffffffff81300000ULL;
	seed_analysis_value(map_fd, 4321, unknown_address);
	system_analysis_compat_value result = lookup_analysis(ctx, map_fd);
	REQUIRE(result.pid == 4321);
	REQUIRE(result.address == unknown_address);
}

TEST_CASE("kernel mode does not complete analysis in userspace",
	  "[syscall-server][system-analysis]")
{
	REQUIRE(setenv("BPFTIME_RUN_WITH_KERNEL", "1", 1) == 0);
	shm_test_fixture fixture;
	syscall_context ctx;

	const int map_fd = create_analysis_map("system_analysis");
	seed_analysis_value(
		map_fd, 4321,
		bpftime::SYNTHETIC_X86_FSBASE_WRITE_TASK_ADDR);
	union bpf_attr attr{};
	uint32_t key = 0;
	system_analysis_compat_value result{};
	attr.map_fd = static_cast<uint32_t>(map_fd);
	attr.key = reinterpret_cast<uint64_t>(&key);
	attr.value = reinterpret_cast<uint64_t>(&result);
	// An shm-only fd cannot be serviced by the real kernel, so passthrough
	// deterministically fails and the seeded value is never rewritten.
	REQUIRE(ctx.handle_sysbpf(BPF_MAP_LOOKUP_ELEM, &attr, sizeof(attr)) <
		0);
	REQUIRE(result.pid == 0);
	unsetenv("BPFTIME_RUN_WITH_KERNEL");
}
#endif
