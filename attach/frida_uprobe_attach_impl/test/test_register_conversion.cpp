#include "catch2/catch_test_macros.hpp"
#include <frida-gum.h>
#include <frida_register_conversion.hpp>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
TEST_CASE("Frida x86 register conversion reports a user-mode pt_regs")
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

	// GumX64CpuContext has no value for these fields. They stay deterministic
	// rather than retaining bytes from the caller's stack.
	REQUIRE(regs.orig_ax == 0);
	REQUIRE(regs.flags == 0);

	// A uprobe fires in user space: synthesize the Linux user-mode selectors
	// (__USER_CS=0x33, __USER_DS=0x2b) so ptregs_is_usermode() succeeds.
	REQUIRE(regs.cs == 0x33);
	REQUIRE(regs.ss == 0x2b);
}
#elif defined(__aarch64__) || defined(_M_ARM64)
TEST_CASE("Frida arm64 register conversion reports an EL0t pt_regs")
{
	_GumArm64CpuContext context{};
	context.x[0] = 0x11;
	context.lr = 0x22;
	context.sp = 0x33;
	context.pc = 0x44;
	// Set only the N/Z/C/V condition flags (bits 31..28); the mode field must
	// be forced to PSR_MODE_EL0t (0) regardless of any low-bit input.
	context.nzcv = 0xf0000000;

	bpftime::pt_regs regs;
	std::memset(&regs, 0xa5, sizeof(regs));
	bpftime::convert_gum_cpu_context_to_pt_regs(context, regs);

	REQUIRE(regs.regs[0] == context.x[0]);
	REQUIRE(regs.regs[30] == context.lr);
	REQUIRE(regs.sp == context.sp);
	REQUIRE(regs.pc == context.pc);

	// Condition flags retained, low four mode bits forced to EL0t.
	REQUIRE((regs.pstate & 0xf0000000ULL) == 0xf0000000ULL);
	REQUIRE((regs.pstate & 0xfULL) == 0);
}
#endif
