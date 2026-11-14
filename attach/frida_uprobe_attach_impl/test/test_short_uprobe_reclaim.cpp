#include "catch2/catch_test_macros.hpp"
#include "frida_uprobe_attach_impl.hpp"
#include <array>
#include <cstring>
#include <stdexcept>

using namespace bpftime::attach;

#if defined(__linux__) && defined(__x86_64__)
extern "C" __attribute__((naked, noinline, aligned(8))) void
__bpftime_test_empty_marker_target()
{
	asm volatile("ret\n\t"
		     "nop; nop; nop; nop; nop; nop; nop\n\t");
}

TEST_CASE("Unsupported empty markers leave host text unchanged",
	  "[frida][short-uprobe]")
{
	frida_attach_impl impl;
	const auto *target = reinterpret_cast<const unsigned char *>(
		__bpftime_test_empty_marker_target);
	std::array<unsigned char, 8> original;
	std::memcpy(original.data(), target, original.size());
	REQUIRE_THROWS_AS(impl.create_uprobe_at(
		(void *)__bpftime_test_empty_marker_target,
		[](const bpftime::pt_regs &) {}), std::runtime_error);
	REQUIRE(std::memcmp(original.data(), target, original.size()) == 0);
	__bpftime_test_empty_marker_target();
}
#endif
