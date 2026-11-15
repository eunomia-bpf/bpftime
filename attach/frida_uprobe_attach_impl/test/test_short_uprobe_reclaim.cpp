#include "catch2/catch_test_macros.hpp"
#include "frida_uprobe_attach_impl.hpp"
#include <array>
#include <cstring>
#include <stdexcept>

using namespace bpftime::attach;

#if defined(__linux__) && defined(__x86_64__)
// The ret-first (0xc3) empty-marker target is defined in hand-written
// assembly (empty_marker_x86_64.S) so its bytes are independent of compiler
// CET instrumentation; see that file for why a compiled naked function is not
// reliable across GCC point releases.
extern "C" void __bpftime_test_empty_marker_target();

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
