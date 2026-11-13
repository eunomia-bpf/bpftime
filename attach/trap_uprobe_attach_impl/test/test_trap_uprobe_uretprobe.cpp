#include "trap_test_common.hpp"
#include "trap_attach_utils.hpp"
#include <trap_uprobe_attach_impl.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <cstdint>

using namespace bpftime;
using namespace bpftime::attach::trap;

extern "C" TRAP_TEST_TARGET uint64_t
__trap_test_simple_add(uint64_t a, uint64_t b)
{
	asm("");
	return a * 2 + b;
}

TEST_CASE("Trap backend: uprobe attach and revert")
{
	int invoke_times = 0;
	uint64_t a = 0, b = 0, a2 = 0, b2 = 0, ret;
	trap_attach_impl man;
	auto func_addr = find_function_addr_by_name("__trap_test_simple_add");
	REQUIRE(func_addr != nullptr);
	REQUIRE(func_addr == (void *)&__trap_test_simple_add);

	int id1 = man.create_uprobe_at(func_addr, [&](const pt_regs &regs) {
		invoke_times++;
		a = PT_REGS_PARM1(&regs);
		b = PT_REGS_PARM2(&regs);
	});
	REQUIRE(id1 >= 0);
	ret = __trap_test_simple_add(2333, 6666);
	REQUIRE(invoke_times == 1);
	REQUIRE(a == 2333);
	REQUIRE(b == 6666);
	REQUIRE(ret == 2333 * 2 + 6666);
	invoke_times = 0;
	a = b = 0;

	int id2 = man.create_uprobe_at(func_addr, [&](const pt_regs &regs) {
		invoke_times++;
		a2 = PT_REGS_PARM1(&regs);
		b2 = PT_REGS_PARM2(&regs);
	});
	REQUIRE(id2 >= 0);
	REQUIRE(id2 != id1);
	ret = __trap_test_simple_add(2333, 6666);
	REQUIRE(invoke_times == 2);
	REQUIRE(a == 2333);
	REQUIRE(b == 6666);
	REQUIRE(a2 == 2333);
	REQUIRE(b2 == 6666);
	REQUIRE(ret == 2333 * 2 + 6666);

	SECTION("Revert by id")
	{
		REQUIRE(man.detach_by_id(id1) == 0);
		REQUIRE(man.detach_by_id(id2) == 0);
		REQUIRE(man.detach_by_id(id2) < 0);
	}
	SECTION("Revert by function address")
	{
		REQUIRE(man.detach_by_func_addr(func_addr) >= 0);
		REQUIRE(man.detach_by_func_addr(func_addr) < 0);
	}
	invoke_times = 0;
	a = b = a2 = b2 = 0;
	ret = __trap_test_simple_add(2333, 6666);
	REQUIRE(ret == 2333 * 2 + 6666);
	REQUIRE(invoke_times == 0);
	REQUIRE(a == 0);
	REQUIRE(a2 == 0);
}

TEST_CASE("Trap backend: uretprobe attach and revert")
{
	int invoke_times = 0;
	uint64_t ret1 = 0, ret2 = 0, dummy;
	trap_attach_impl man;
	auto func_addr = find_function_addr_by_name("__trap_test_simple_add");
	REQUIRE(func_addr != nullptr);

	int id1 = man.create_uretprobe_at(func_addr, [&](const pt_regs &regs) {
		invoke_times++;
		ret1 = PT_REGS_RC(&regs);
	});
	REQUIRE(id1 >= 0);
	dummy = __trap_test_simple_add(2333, 6666);
	REQUIRE(invoke_times == 1);
	REQUIRE(dummy == 2333 * 2 + 6666);
	REQUIRE(ret1 == dummy);
	invoke_times = 0;
	ret1 = 0;

	int id2 = man.create_uretprobe_at(func_addr, [&](const pt_regs &regs) {
		invoke_times++;
		ret2 = PT_REGS_RC(&regs);
	});
	REQUIRE(id2 >= 0);
	dummy = __trap_test_simple_add(2333, 6666);
	REQUIRE(invoke_times == 2);
	REQUIRE(ret1 == 2333 * 2 + 6666);
	REQUIRE(ret2 == 2333 * 2 + 6666);

	SECTION("Revert by id")
	{
		REQUIRE(man.detach_by_id(id1) == 0);
		REQUIRE(man.detach_by_id(id2) == 0);
	}
	SECTION("Revert by function address")
	{
		REQUIRE(man.detach_by_func_addr(func_addr) >= 0);
	}
	invoke_times = 0;
	ret1 = ret2 = 0;
	dummy = __trap_test_simple_add(2333, 6666);
	REQUIRE(dummy == 2333 * 2 + 6666);
	REQUIRE(ret1 == 0);
	REQUIRE(ret2 == 0);
	REQUIRE(invoke_times == 0);
}

TEST_CASE("Trap backend: mixed uprobe and uretprobe")
{
	using namespace Catch::Generators;
	int uprobe_invoke_times = 0;
	int uretprobe_invoke_times = 0;
	uint64_t a = 0, b = 0, ret = 0;
	trap_attach_impl man;
	auto func_addr = find_function_addr_by_name("__trap_test_simple_add");
	REQUIRE(func_addr != nullptr);
	int id1 = man.create_uprobe_at(func_addr, [&](const pt_regs &regs) {
		uprobe_invoke_times++;
		a = PT_REGS_PARM1(&regs);
		b = PT_REGS_PARM2(&regs);
	});
	REQUIRE(id1 >= 0);
	int id2 = man.create_uretprobe_at(func_addr, [&](const pt_regs &regs) {
		uretprobe_invoke_times++;
		ret = PT_REGS_RC(&regs);
	});
	REQUIRE(id2 >= 0);
	uint64_t i = GENERATE(take(10, random(0, 1000)));
	uint64_t j = GENERATE(take(10, random(0, 1000)));
	uint64_t expected = i * 2 + j;
	uint64_t dummy = __trap_test_simple_add(i, j);
	REQUIRE(dummy == expected);
	REQUIRE(uprobe_invoke_times == 1);
	REQUIRE(uretprobe_invoke_times == 1);
	REQUIRE(a == i);
	REQUIRE(b == j);
	REQUIRE(ret == dummy);
	int count = 0;
	man.iterate_attaches([&](int id, const void *addr, int ty) {
		count++;
		REQUIRE(addr == func_addr);
		REQUIRE((id == id1 || id == id2));
		REQUIRE((ty == ATTACH_UPROBE || ty == ATTACH_URETPROBE));
	});
	REQUIRE(count == 2);
}

extern "C" uint64_t __trap_test_recursive(uint64_t n);
// Recurse through a volatile pointer so that the optimizer cannot turn the
// recursion into a loop (which would make the function enter only once).
static uint64_t (*volatile __trap_test_recursive_ptr)(uint64_t) =
	__trap_test_recursive;

extern "C" TRAP_TEST_TARGET uint64_t __trap_test_recursive(uint64_t n)
{
	asm("");
	if (n == 0)
		return 0;
	return n + __trap_test_recursive_ptr(n - 1);
}

TEST_CASE("Trap backend: uretprobe on a recursive function")
{
	trap_attach_impl man;
	int entries = 0, exits = 0;
	uint64_t last_ret = 0;
	auto addr = (void *)&__trap_test_recursive;
	REQUIRE(man.create_uprobe_at(addr, [&](const pt_regs &) {
		entries++;
	}) >= 0);
	REQUIRE(man.create_uretprobe_at(addr, [&](const pt_regs &regs) {
		exits++;
		last_ret = PT_REGS_RC(&regs);
	}) >= 0);
	REQUIRE(__trap_test_recursive(20) == 210);
	REQUIRE(entries == 21);
	REQUIRE(exits == 21);
	REQUIRE(last_ret == 210);
}

TEST_CASE("Trap backend: instances are independent")
{
	int count_a = 0, count_b = 0;
	auto addr = (void *)&__trap_test_simple_add;
	trap_attach_impl a;
	REQUIRE(a.create_uprobe_at(addr, [&](const pt_regs &) { count_a++; }) >=
		0);
	{
		trap_attach_impl b;
		REQUIRE(b.create_uprobe_at(addr, [&](const pt_regs &) {
			count_b++;
		}) >= 0);
		__trap_test_simple_add(1, 2);
		REQUIRE(count_a == 1);
		REQUIRE(count_b == 1);
	}
	// b's probe is gone, a's still fires
	__trap_test_simple_add(1, 2);
	REQUIRE(count_a == 2);
	REQUIRE(count_b == 1);
}
