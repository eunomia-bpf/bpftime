#include "frida_attach_utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <frida_uprobe_attach_impl.hpp>
#include <spdlog/spdlog.h>
#if !defined(__x86_64__) && defined(_M_X64)
#error Only supports x86_64
#endif
using namespace bpftime::attach;
using namespace bpftime;

extern "C" uint64_t bpftime_set_retval(uint64_t retval);
extern "C" __attribute__((__noinline__)) uint64_t
__bpftime_func_to_filter(uint64_t a, uint64_t b)
{ // Forbid inline
	asm("");
	return (a << 32) | b;
}
__attribute__((__noinline__, noinline)) static uint64_t
call_filter_func(uint64_t a, uint64_t b)
{
	return __bpftime_func_to_filter(a, b);
}

TEST_CASE("Test attaching filter programs and revert")
{
	frida_attach_impl man;
	auto func_addr =
		find_function_addr_by_name("__bpftime_func_to_filter");

	REQUIRE(func_addr != nullptr);
	const uint64_t a = 0xabce;
	const uint64_t b = 0x1234;
	const uint64_t expected_result = (a << 32) | b;
	REQUIRE(__bpftime_func_to_filter(a, b) == expected_result);
	int id = man.create_uprobe_override_at(
		func_addr, [&](const pt_regs &regs) {
			uint64_t first_arg = PT_REGS_PARM1(&regs);
			uint64_t second_arg = PT_REGS_PARM2(&regs);
			if (first_arg == a) {
				bpftime_set_retval(first_arg + second_arg);
			}
		});
	REQUIRE(id >= 0);
	REQUIRE(__bpftime_func_to_filter(1, 2) == (((uint64_t)1 << 32) | 2));
	REQUIRE(call_filter_func(a, b) == a + b);

	// Revert it
	SECTION("Revert by id")
	{
		REQUIRE(man.detach_by_id(id) >= 0);
	}
	SECTION("Revert by function address")
	{
		REQUIRE(man.detach_by_func_addr(func_addr) >= 0);
	}

	REQUIRE(__bpftime_func_to_filter(1, 2) == (((uint64_t)1 << 32) | 2));
	REQUIRE(__bpftime_func_to_filter(a, b) == expected_result);
}
