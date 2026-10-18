#include "trap_test_common.hpp"
#include "trap_attach_utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <trap_uprobe_attach_impl.hpp>

using namespace bpftime;
using namespace bpftime::attach::trap;

extern "C" uint64_t bpftime_set_retval(uint64_t retval);

extern "C" TRAP_TEST_TARGET uint64_t
__trap_func_to_filter(uint64_t a, uint64_t b)
{
	asm("");
	return (a << 32) | b;
}

TRAP_TEST_TARGET static uint64_t call_filter_func(uint64_t a,
							       uint64_t b)
{
	return __trap_func_to_filter(a, b);
}

TEST_CASE("Trap backend: filter attach (override return) and revert")
{
	trap_attach_impl man;
	auto func_addr = find_function_addr_by_name("__trap_func_to_filter");
	REQUIRE(func_addr != nullptr);
	const uint64_t a = 0xabce;
	const uint64_t b = 0x1234;
	const uint64_t expected = (a << 32) | b;
	REQUIRE(__trap_func_to_filter(a, b) == expected);

	int id = man.create_uprobe_override_at(
		func_addr, [&](const pt_regs &regs) {
			uint64_t first = PT_REGS_PARM1(&regs);
			uint64_t second = PT_REGS_PARM2(&regs);
			if (first == a)
				bpftime_set_retval(first + second);
		});
	REQUIRE(id >= 0);
	// Not overridden: original body runs
	REQUIRE(__trap_func_to_filter(1, 2) == (((uint64_t)1 << 32) | 2));
	// Overridden: body skipped, our value returned to the caller
	REQUIRE(call_filter_func(a, b) == a + b);
	REQUIRE(__trap_func_to_filter(a, b) == a + b);

	SECTION("Revert by id")
	{
		REQUIRE(man.detach_by_id(id) >= 0);
	}
	SECTION("Revert by function address")
	{
		REQUIRE(man.detach_by_func_addr(func_addr) >= 0);
	}
	REQUIRE(__trap_func_to_filter(1, 2) == (((uint64_t)1 << 32) | 2));
	REQUIRE(__trap_func_to_filter(a, b) == expected);
}

TEST_CASE("Trap backend: replace attach through the ebpf callback interface")
{
	trap_attach_impl man;
	auto func_addr = (void *)&__trap_func_to_filter;
	int id = man.attach_at_with_ebpf_callback(
		func_addr,
		ebpf_callback_args{
			.ebpf_cb = [](void *mem, size_t, uint64_t *ret) -> int {
				auto *regs = (pt_regs *)mem;
				*ret = PT_REGS_PARM1(regs) * 1000 +
				       PT_REGS_PARM2(regs);
				bpftime_set_retval(*ret);
				return 0;
			},
			.attach_type = ATTACH_UPROBE_OVERRIDE });
	REQUIRE(id >= 0);
	REQUIRE(__trap_func_to_filter(7, 9) == 7009);
	REQUIRE(man.detach_by_id(id) == 0);
	REQUIRE(__trap_func_to_filter(7, 9) == (((uint64_t)7 << 32) | 9));
}

TEST_CASE("Trap backend: override and probes cannot share a function")
{
	trap_attach_impl man;
	auto func_addr = (void *)&__trap_func_to_filter;
	int id = man.create_uprobe_override_at(func_addr,
					       [](const pt_regs &) {});
	REQUIRE(id >= 0);
	REQUIRE(man.create_uprobe_at(func_addr, [](const pt_regs &) {}) ==
		-EEXIST);
	REQUIRE(man.create_uretprobe_at(func_addr, [](const pt_regs &) {}) ==
		-EEXIST);
	REQUIRE(man.detach_by_id(id) == 0);

	int uid = man.create_uprobe_at(func_addr, [](const pt_regs &) {});
	REQUIRE(uid >= 0);
	REQUIRE(man.create_uprobe_override_at(func_addr,
					      [](const pt_regs &) {}) ==
		-EEXIST);
	REQUIRE(man.detach_by_id(uid) == 0);
	REQUIRE(__trap_func_to_filter(1, 1) == (((uint64_t)1 << 32) | 1));
}
