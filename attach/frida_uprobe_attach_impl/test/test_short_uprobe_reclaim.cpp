#include "catch2/catch_test_macros.hpp"
#include "frida_uprobe_attach_impl.hpp"
#include <array>
#include <cstring>
#include <stdexcept>

using namespace bpftime::attach;

#if defined(__linux__) && defined(__x86_64__)
// This marker must stay ret-first (first byte 0xc3) so Frida-Gum rejects it.
// CET-enabled toolchains (e.g. Fedora GCC 13, Ubuntu GCC 12, Clang with
// -fcf-protection=full) otherwise prefix endbr64, which Frida-Gum accepts,
// breaking the rejection assertion. CET is disabled for this translation unit
// via a scoped -fcf-protection=none in this directory's CMakeLists.txt; that
// is deterministic across GCC/Clang point releases, whereas a per-symbol
// nocf_check attribute is not honored uniformly on naked functions by GCC.
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
