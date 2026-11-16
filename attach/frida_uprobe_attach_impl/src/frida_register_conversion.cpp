#include <frida_register_conversion.hpp>
#include <frida-gum.h>
namespace bpftime
{
#if defined(__x86_64__) || defined(_M_X64)
namespace
{
// Linux user-mode segment selectors (arch/x86/include/asm/segment.h):
// __USER_CS / __USER_DS carry Requested Privilege Level 3. A userspace probe
// context must report these so an eBPF program's user-mode check, such as
// ptregs_is_usermode() comparing regs->ss against __USER_DS, classifies the
// context correctly instead of entering kernel-only helper paths that do not
// exist in a userspace runtime.
constexpr uint64_t X86_USER_CS = 0x33;
constexpr uint64_t X86_USER_DS = 0x2b;
} // namespace

void convert_gum_cpu_context_to_pt_regs(const ::_GumX64CpuContext &context,
					pt_regs &regs)
{
	// Gum only exposes the general-purpose register subset. Clear fields such
	// as orig_ax, cs, flags, and ss instead of leaking uninitialized stack data
	// into the BPF pt_regs context.
	regs = {};
	regs.ip = context.rip;
	regs.r15 = context.r15;
	regs.r14 = context.r14;
	regs.r13 = context.r13;
	regs.r12 = context.r12;
	regs.r11 = context.r11;
	regs.r10 = context.r10;
	regs.r9 = context.r9;
	regs.r8 = context.r8;
	regs.di = context.rdi;
	regs.si = context.rsi;
	regs.bp = context.rbp;
	regs.sp = context.rsp;
	regs.bx = context.rbx;
	regs.dx = context.rdx;
	regs.cx = context.rcx;
	regs.ax = context.rax;
	// Frida only observes user-space execution, so the interrupted context is
	// always user mode. Synthesize the user selectors instead of leaving them
	// zero so user-mode tests in eBPF programs pass.
	regs.cs = X86_USER_CS;
	regs.ss = X86_USER_DS;
}

void convert_pt_regs_to_gum_cpu_context(const pt_regs &regs,
					::_GumX64CpuContext &context)
{
	context.rip = regs.ip;
	context.r15 = regs.r15;
	context.r14 = regs.r14;
	context.r13 = regs.r13;
	context.r12 = regs.r12;
	context.r11 = regs.r11;
	context.r10 = regs.r10;
	context.r9 = regs.r9;
	context.r8 = regs.r8;
	context.rdi = regs.di;
	context.rsi = regs.si;
	context.rbp = regs.bp;
	context.rsp = regs.sp;
	context.rbx = regs.bx;
	context.rdx = regs.dx;
	context.rcx = regs.cx;
	context.rax = regs.ax;
}

#elif defined(__aarch64__) || defined(_M_ARM64)
namespace
{
// ARM64 program status register mode field (arch/arm64/include/asm/ptrace.h).
// PSR_MODE_EL0t is 0 and PSR_MODE_MASK selects the low four bits; a userspace
// probe context must report EL0 so eBPF user-mode checks classify it the same
// way the kernel pt_regs for a user interrupt would.
constexpr uint64_t ARM64_PSR_MODE_MASK = 0xf;
} // namespace

void convert_gum_cpu_context_to_pt_regs(const ::_GumArm64CpuContext &context,
					pt_regs &regs)
{
	regs = {};
	memcpy(&regs.regs, &context.x, sizeof(context.x));
	regs.regs[29] = context.fp;
	regs.regs[30] = context.lr;
	regs.sp = context.sp;
	regs.pc = context.pc;
	// Keep the condition flags from Gum but force the mode field to EL0t.
	regs.pstate = context.nzcv & ~ARM64_PSR_MODE_MASK;
}

void convert_pt_regs_to_gum_cpu_context(const pt_regs &regs,
					::_GumArm64CpuContext &context)
{
	memcpy(&context.x, &regs.regs, sizeof(context.x));
	context.fp = regs.regs[29];
	context.lr = regs.regs[30];
	context.sp = regs.sp;
	context.pc = regs.pc;
	context.nzcv = regs.pstate;
}
#elif defined(__arm__) || defined(_M_ARM)
void convert_gum_cpu_context_to_pt_regs(const ::_GumArmCpuContext &context,
					pt_regs &regs)
{
	for (size_t i = 0; i < std::size(context.r); i++) {
		regs.uregs[i] = context.r[i];
	}
	regs.uregs[8] = context.r8;
	regs.uregs[9] = context.r9;
	regs.uregs[10] = context.r10;
	regs.uregs[11] = context.r11;
	regs.uregs[12] = context.r12;
	regs.uregs[13] = context.sp;
	regs.uregs[14] = context.lr;
	regs.uregs[15] = context.pc;
	regs.uregs[16] = context.cpsr;
	regs.uregs[17] = 0;
}

void convert_pt_regs_to_gum_cpu_context(const pt_regs &regs,
					::_GumArmCpuContext &context)
{
	for (size_t i = 0; i < std::size(context.r); i++) {
		context.r[i] = regs.uregs[i];
	}
	context.r8 = regs.uregs[8];
	context.r9 = regs.uregs[9];
	context.r10 = regs.uregs[10];
	context.r11 = regs.uregs[11];
	context.r12 = regs.uregs[12];
	context.sp = regs.uregs[13];
	context.lr = regs.uregs[14];
	context.pc = regs.uregs[15];
	context.cpsr = regs.uregs[16];
}
#else
#error "Unsupported architecture"
#endif
} // namespace bpftime
