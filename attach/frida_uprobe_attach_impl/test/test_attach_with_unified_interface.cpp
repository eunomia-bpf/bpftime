#include "base_attach_impl.hpp"
#include "catch2/catch_test_macros.hpp"
#include "frida_attach_private_data.hpp"
#include "frida_uprobe_attach_impl.hpp"
#include "frida_uprobe_attach_internal.hpp"
#include <cstdint>
#include <frida_uprobe.hpp>
#include <memory>
#include <string>

using namespace bpftime::attach;

extern "C" uint64_t
__bpftime_test_attach_with_unified_interface_func(uint64_t a, uint64_t b)
{
	return a + b;
}

TEST_CASE("Test with unified interface")
{
	std::unique_ptr<base_attach_impl> man =
		std::make_unique<frida_attach_impl>();
	frida_attach_private_data priv;
	REQUIRE(priv.initialize_from_string(std::to_string((
			uint64_t)(uintptr_t)(&__bpftime_test_attach_with_unified_interface_func))) ==
		0);
	REQUIRE(priv.addr ==
		(uintptr_t)&__bpftime_test_attach_with_unified_interface_func);
	int id;
	bool invoked = false;
	SECTION("uprobe")
	{
		id = man->handle_attach_with_ebpf_call_back(
			[&](const void *v, size_t, uint64_t *) {
				bpftime::pt_regs *mem = (bpftime::pt_regs *)v;
				REQUIRE(mem->di == 1);
				REQUIRE(mem->si == 2);
				invoked = true;
				return 0;
			},
			priv, ATTACH_UPROBE);
	}
	SECTION("uretprobe")
	{
		id = man->handle_attach_with_ebpf_call_back(
			[&](const void *v, size_t, uint64_t *) {
				bpftime::pt_regs *mem = (bpftime::pt_regs *)v;
				REQUIRE(mem->ax == 3);
				invoked = true;
				return 0;
			},
			priv, ATTACH_URETPROBE);
	}
	REQUIRE(id >= 0);
	REQUIRE(__bpftime_test_attach_with_unified_interface_func(1, 2) ==
		1 + 2);
	REQUIRE(invoked == true);
	REQUIRE(man->detach_by_id(id) >= 0);
	invoked = false;
	REQUIRE(__bpftime_test_attach_with_unified_interface_func(1, 2) ==
		1 + 2);
	REQUIRE(invoked == false);
}
