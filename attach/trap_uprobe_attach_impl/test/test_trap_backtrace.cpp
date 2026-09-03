#include "trap_test_common.hpp"
#include <trap_uprobe_attach_impl.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <vector>

using namespace bpftime;
using namespace bpftime::attach::trap;

extern "C" TRAP_TEST_TARGET uint64_t __trap_bt_leaf(uint64_t a)
{
	asm("");
	return a * 7;
}

extern "C" TRAP_TEST_TARGET uint64_t __trap_bt_caller(uint64_t a)
{
	asm("");
	uint64_t r = __trap_bt_leaf(a) + 1;
	asm("");
	return r;
}

static bool within(uint64_t pc, const void *func, size_t span = 4096)
{
	return pc > (uintptr_t)func && pc < (uintptr_t)func + span;
}

TEST_CASE("Trap backend: generate_stack reports the callers of the probe")
{
	trap_attach_impl man;
	std::vector<uint64_t> entry_stack, exit_stack;
	REQUIRE(man.create_uprobe_at((void *)&__trap_bt_leaf,
				     [&](const pt_regs &) {
					     auto *p = (std::vector<uint64_t> *)
							       man.call_attach_specific_function(
								       "generate_stack",
								       nullptr);
					     REQUIRE(p != nullptr);
					     entry_stack = *p;
					     delete p;
				     }) >= 0);
	REQUIRE(man.create_uretprobe_at((void *)&__trap_bt_leaf,
					[&](const pt_regs &) {
						auto *p = (std::vector<uint64_t> *)
								  man.call_attach_specific_function(
									  "generate_stack",
									  nullptr);
						REQUIRE(p != nullptr);
						exit_stack = *p;
						delete p;
					}) >= 0);
	REQUIRE(__trap_bt_caller(3) == 22);
	REQUIRE(!entry_stack.empty());
	REQUIRE(!exit_stack.empty());
	// The first frame after the probed function is the return address
	// inside __trap_bt_caller
	REQUIRE(within(entry_stack[0], (const void *)&__trap_bt_caller));
	bool found = false;
	for (auto f : exit_stack)
		found |= within(f, (const void *)&__trap_bt_caller);
	REQUIRE(found);
	// Outside of a probe there is no stack to report
	REQUIRE(man.call_attach_specific_function("generate_stack", nullptr) ==
		nullptr);
	REQUIRE(man.call_attach_specific_function("no_such_feature", nullptr) ==
		nullptr);
}
