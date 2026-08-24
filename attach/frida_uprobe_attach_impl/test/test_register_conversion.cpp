#include "catch2/catch_test_macros.hpp"
#include <frida-gum.h>
#include <frida_register_conversion.hpp>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
TEST_CASE("Frida x86 register conversion clears unavailable pt_regs fields")
{
	_GumX64CpuContext context{};
	context.rax = 0x11;
	context.rdi = 0x22;
	context.rip = 0x33;
	context.rsp = 0x44;

	bpftime::pt_regs regs;
	std::memset(&regs, 0xa5, sizeof(regs));
	bpftime::convert_gum_cpu_context_to_pt_regs(context, regs);

	REQUIRE(regs.ax == context.rax);
	REQUIRE(regs.di == context.rdi);
	REQUIRE(regs.ip == context.rip);
	REQUIRE(regs.sp == context.rsp);

	// GumX64CpuContext has no values for these Linux pt_regs fields. They must
	// be deterministic instead of retaining bytes from the caller's stack.
	REQUIRE(regs.orig_ax == 0);
	REQUIRE(regs.cs == 0);
	REQUIRE(regs.flags == 0);
	REQUIRE(regs.ss == 0);
}
#endif
