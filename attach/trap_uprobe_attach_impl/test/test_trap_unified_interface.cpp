#include "trap_test_common.hpp"
#include "base_attach_impl.hpp"
#include "trap_attach_private_data.hpp"
#include "trap_uprobe_attach_impl.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <string>

using namespace bpftime::attach;
using namespace bpftime::attach::trap;

extern "C" TRAP_TEST_TARGET uint64_t
__trap_test_unified_func(uint64_t a, uint64_t b)
{
	asm("");
	return a + b;
}

TEST_CASE("Trap backend: unified base_attach_impl interface")
{
	std::unique_ptr<base_attach_impl> man =
		std::make_unique<trap_attach_impl>();
	trap_attach_private_data priv;
	REQUIRE(priv.initialize_from_string(std::to_string(
			(uint64_t)(uintptr_t)&__trap_test_unified_func)) == 0);
	REQUIRE(priv.addr == (uintptr_t)&__trap_test_unified_func);
	int id = -1;
	bool invoked = false;
	SECTION("uprobe")
	{
		id = man->create_attach_with_ebpf_callback(
			[&](void *v, size_t sz, uint64_t *) {
				auto *mem = (bpftime::pt_regs *)v;
				REQUIRE(sz == sizeof(bpftime::pt_regs));
				REQUIRE(PT_REGS_PARM1(mem) == 1);
				REQUIRE(PT_REGS_PARM2(mem) == 2);
				invoked = true;
				return 0;
			},
			priv, ATTACH_UPROBE);
		REQUIRE(id >= 0);
		REQUIRE(__trap_test_unified_func(1, 2) == 3);
	}
	SECTION("uretprobe")
	{
		id = man->create_attach_with_ebpf_callback(
			[&](void *v, size_t, uint64_t *) {
				auto *mem = (bpftime::pt_regs *)v;
				REQUIRE(PT_REGS_RC(mem) == 3);
				invoked = true;
				return 0;
			},
			priv, ATTACH_URETPROBE);
		REQUIRE(id >= 0);
		REQUIRE(__trap_test_unified_func(1, 2) == 3);
	}
	SECTION("ureplace")
	{
		id = man->create_attach_with_ebpf_callback(
			[&](void *, size_t, uint64_t *ret) {
				invoked = true;
				*ret = 42;
				return 0;
			},
			priv, ATTACH_UREPLACE);
		REQUIRE(id >= 0);
		REQUIRE(__trap_test_unified_func(1, 2) == 42);
	}
	REQUIRE(invoked == true);
	REQUIRE(man->detach_by_id(id) >= 0);
	invoked = false;
	REQUIRE(__trap_test_unified_func(1, 2) == 3);
	REQUIRE(invoked == false);
}

TEST_CASE("Trap backend: rejects foreign private data and bad attach types")
{
	trap_attach_impl man;
	struct other_private_data : public attach_private_data {
	} other;
	REQUIRE(man.create_attach_with_ebpf_callback(
			[](void *, size_t, uint64_t *) { return 0; }, other,
			ATTACH_UPROBE) == -EINVAL);
	trap_attach_private_data priv;
	REQUIRE(priv.initialize_from_string(std::to_string(
			(uint64_t)(uintptr_t)&__trap_test_unified_func)) == 0);
	REQUIRE(man.create_attach_with_ebpf_callback(
			[](void *, size_t, uint64_t *) { return 0; }, priv,
			12345) == -ENOTSUP);
	REQUIRE(man.create_uprobe_at(nullptr, [](const bpftime::pt_regs &) {}) ==
		-EINVAL);
}
