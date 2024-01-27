/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _ATTACH_INTERNAL_HPP
#define _ATTACH_INTERNAL_HPP
#include <frida-gum.h>
namespace bpftime
{

#if defined(__x86_64__) || defined(_M_X64)

struct pt_regs {
	uint64_t r15;
	uint64_t r14;
	uint64_t r13;
	uint64_t r12;
	uint64_t bp;
	uint64_t bx;
	uint64_t r11;
	uint64_t r10;
	uint64_t r9;
	uint64_t r8;
	uint64_t ax;
	uint64_t cx;
	uint64_t dx;
	uint64_t si;
	uint64_t di;
	uint64_t orig_ax;
	uint64_t ip;
	uint64_t cs;
	uint64_t flags;
	uint64_t sp;
	uint64_t ss;
};
// https://github.com/torvalds/linux/blob/6613476e225e090cc9aad49be7fa504e290dd33d/tools/lib/bpf/bpf_tracing.h#L79
#define PT_REGS_PARM1(x) ((x)->di)
#define PT_REGS_PARM2(x) ((x)->si)
#define PT_REGS_PARM3(x) ((x)->dx)
#define PT_REGS_PARM4(x) ((x)->cx)
#define PT_REGS_PARM5(x) ((x)->r8)
#define PT_REGS_PARM6(x) ((x)->r9)
#define PT_REGS_RET(x) ((x)->sp)
#define PT_REGS_RC(x) ((x)->ax)

#elif defined(__aarch64__) || defined(_M_ARM64)
struct pt_regs {
	uint64_t regs[31];
	uint64_t sp;
	uint64_t pc;
	uint64_t pstate;
};
// https://github.com/torvalds/linux/blob/6613476e225e090cc9aad49be7fa504e290dd33d/tools/lib/bpf/bpf_tracing.h#L217
#define PT_REGS_PARM1(x) ((x)->regs[0])
#define PT_REGS_PARM2(x) ((x)->regs[1])
#define PT_REGS_PARM3(x) ((x)->regs[2])
#define PT_REGS_PARM4(x) ((x)->regs[3])
#define PT_REGS_PARM5(x) ((x)->regs[4])
#define PT_REGS_PARM6(x) ((x)->regs[5])
#define PT_REGS_PARM7(x) ((x)->regs[6])
#define PT_REGS_PARM8(x) ((x)->regs[7])
#define PT_REGS_RET(x) ((x)->regs[30])
#define PT_REGS_RC(x) ((x)->regs[0])

#elif defined(__arm__) || defined(_M_ARM)
// https://github.com/torvalds/linux/blob/6613476e225e090cc9aad49be7fa504e290dd33d/tools/lib/bpf/bpf_tracing.h#L192
struct pt_regs {
	uint32_t uregs[18];
};
#define PT_REGS_PARM1(x) ((x)->uregs[0])
#define PT_REGS_PARM2(x) ((x)->uregs[1])
#define PT_REGS_PARM3(x) ((x)->uregs[2])
#define PT_REGS_PARM4(x) ((x)->uregs[3])
#define PT_REGS_RET(x) ((x)->uregs[14])
#define PT_REGS_RC(x) ((x)->uregs[0])
#else
#error "Unsupported architecture"
#endif


#if defined(__x86_64__) || defined(_M_X64)

static inline void
convert_gum_cpu_context_to_pt_regs(const _GumX64CpuContext &context,
				   pt_regs &regs)
{
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
}

static inline void
convert_pt_regs_to_gum_cpu_context(const pt_regs &regs,
				   _GumX64CpuContext &context)
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
static inline void
convert_gum_cpu_context_to_pt_regs(const _GumArm64CpuContext &context,
				   pt_regs &regs)
{
	memcpy(&regs.regs, &context.x, sizeof(context.x));
	regs.regs[29] = context.fp;
	regs.regs[30] = context.lr;
	regs.sp = context.sp;
	regs.pc = context.pc;
	regs.pstate = context.nzcv;
}

static inline void
convert_pt_regs_to_gum_cpu_context(const pt_regs &regs,
				   _GumArm64CpuContext &context)
{
	memcpy(&context.x, &regs.regs, sizeof(context.x));
	context.fp = regs.regs[29];
	context.lr = regs.regs[30];
	context.sp = regs.sp;
	context.pc = regs.pc;
	context.nzcv = regs.pstate;
}
#elif defined(__arm__) || defined(_M_ARM)
static inline void
convert_gum_cpu_context_to_pt_regs(const _GumArmCpuContext &context,
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

static inline void
convert_pt_regs_to_gum_cpu_context(const pt_regs &regs,
				   _GumArmCpuContext &context)
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
// GType uprobe_listener_get_type();
} // namespace bpftime
#endif
