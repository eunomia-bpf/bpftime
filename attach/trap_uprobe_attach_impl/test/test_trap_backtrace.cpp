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

TEST_CASE("Trap backend: allocating stack collection is unavailable")
{
	trap_attach_impl man;
	const std::string feature = "generate_stack";
	bool entry_rejected = false, return_rejected = false;
	REQUIRE(man.create_uprobe_at((void *)&__trap_bt_leaf,
		[&](const pt_regs &) {
			entry_rejected = man.call_attach_specific_function(feature, nullptr) == nullptr;
		}) >= 0);
	REQUIRE(man.create_uretprobe_at((void *)&__trap_bt_leaf,
		[&](const pt_regs &) {
			return_rejected = man.call_attach_specific_function(feature, nullptr) == nullptr;
		}) >= 0);
	REQUIRE(__trap_bt_caller(3) == 22);
	REQUIRE(entry_rejected);
	REQUIRE(return_rejected);
	REQUIRE(man.call_attach_specific_function(feature, nullptr) == nullptr);
}
