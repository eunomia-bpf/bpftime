#include "frida_attach_utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <frida_uprobe_attach_impl.hpp>
#include <spdlog/spdlog.h>
#if !defined(__x86_64__) && defined(_M_X64)
#error Only supports x86_64
#endif

using namespace bpftime::attach;
extern "C" __attribute__((__noinline__, noinline)) uint64_t
__bpftime_func_to_replace(uint64_t a, uint64_t b)
{
	// Forbid inline
	asm("");
	return (a << 32) | b;
}
__attribute__((__noinline__, noinline)) static uint64_t
call_replace_func(uint64_t a, uint64_t b)
{
	return __bpftime_func_to_replace(a, b);
}
extern "C" uint64_t bpftime_set_retval(uint64_t value);
TEST_CASE("Test attaching replace programs and revert")
{
	frida_attach_impl man;
	auto func_addr =
		find_function_addr_by_name("__bpftime_func_to_replace");

	REQUIRE(func_addr != nullptr);
	const uint64_t a = 0xabce;
	const uint64_t b = 0x1234;
	const uint64_t expected_result = (a << 32) | b;
	REQUIRE(__bpftime_func_to_replace(a, b) == expected_result);
	int invoke_times = 0;
	int id = man.create_uprobe_override_at(
		func_addr, [&](const bpftime::pt_regs &regs) {
			invoke_times++;
			bpftime_set_retval(PT_REGS_PARM1(&regs) +
					   PT_REGS_PARM2(&regs));
		});
	REQUIRE(id >= 0);
	REQUIRE(call_replace_func(a, b) == a + b);
	REQUIRE(invoke_times == 1);
	// Revert it
	invoke_times = 0;
	SECTION("Revert by id")
	{
		REQUIRE(man.detach_by_id(id) >= 0);
	}
	SECTION("Revert by function address")
	{
		REQUIRE(man.detach_by_func_addr(func_addr) >= 0);
	}
	REQUIRE(__bpftime_func_to_replace(a, b) == expected_result);
	REQUIRE(invoke_times == 0);
}
