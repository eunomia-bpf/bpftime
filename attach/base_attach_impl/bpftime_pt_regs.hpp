/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
// Architecture-neutral definition of the `pt_regs` structure handed to eBPF
// programs by uprobe-style attach implementations.
//
// The layouts and PT_REGS_* accessors mirror the kernel's
// tools/lib/bpf/bpf_tracing.h so that an eBPF program compiled against
// vmlinux.h reads the same offsets in userspace.
//
// NOTE: the include guard is deliberately shared with
// attach/frida_uprobe_attach_impl/include/frida_register_def.hpp, which
// defines identical x86_64 / aarch64 / arm layouts. Whichever header is
// included first wins, so translation units that pull in both the frida and
// the trap attach implementations do not see a duplicate definition.
#ifndef _BPFTIME_FRIDA_PT_REGS
#define _BPFTIME_FRIDA_PT_REGS
#include <cstdint>
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

#elif defined(__riscv) && __riscv_xlen == 64
// Layout of struct pt_regs in arch/riscv/include/asm/ptrace.h. The field
// order matches the __gregs[] array of the signal ucontext (index 0 is pc).
struct pt_regs {
	uint64_t epc;
	uint64_t ra;
	uint64_t sp;
	uint64_t gp;
	uint64_t tp;
	uint64_t t0;
	uint64_t t1;
	uint64_t t2;
	uint64_t s0;
	uint64_t s1;
	uint64_t a0;
	uint64_t a1;
	uint64_t a2;
	uint64_t a3;
	uint64_t a4;
	uint64_t a5;
	uint64_t a6;
	uint64_t a7;
	uint64_t s2;
	uint64_t s3;
	uint64_t s4;
	uint64_t s5;
	uint64_t s6;
	uint64_t s7;
	uint64_t s8;
	uint64_t s9;
	uint64_t s10;
	uint64_t s11;
	uint64_t t3;
	uint64_t t4;
	uint64_t t5;
	uint64_t t6;
};
#define PT_REGS_PARM1(x) ((x)->a0)
#define PT_REGS_PARM2(x) ((x)->a1)
#define PT_REGS_PARM3(x) ((x)->a2)
#define PT_REGS_PARM4(x) ((x)->a3)
#define PT_REGS_PARM5(x) ((x)->a4)
#define PT_REGS_PARM6(x) ((x)->a5)
#define PT_REGS_PARM7(x) ((x)->a6)
#define PT_REGS_PARM8(x) ((x)->a7)
#define PT_REGS_RET(x) ((x)->ra)
#define PT_REGS_RC(x) ((x)->a0)

#elif defined(__arm__) || defined(_M_ARM)
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

} // namespace bpftime

#endif
