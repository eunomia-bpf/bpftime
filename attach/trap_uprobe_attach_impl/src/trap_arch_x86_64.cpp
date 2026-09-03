/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#if defined(__x86_64__) || defined(_M_X64)
#include "trap_arch.hpp"
#include "x86_insn_decode.hpp"
#include <cstring>
#include <sys/ucontext.h>

namespace bpftime::attach::trap::arch
{
namespace
{
constexpr uint8_t INT3 = 0xcc;

inline greg_t &reg(ucontext_t *uc, int idx)
{
	return uc->uc_mcontext.gregs[idx];
}
inline greg_t reg(const ucontext_t *uc, int idx)
{
	return uc->uc_mcontext.gregs[idx];
}

// Evaluate a Jcc condition code against EFLAGS
bool eval_condition(unsigned cc, uint64_t flags)
{
	const bool cf = flags & (1 << 0);
	const bool pf = flags & (1 << 2);
	const bool zf = flags & (1 << 6);
	const bool sf = flags & (1 << 7);
	const bool of = flags & (1 << 11);
	bool result;
	switch (cc >> 1) {
	case 0:
		result = of;
		break;
	case 1:
		result = cf;
		break;
	case 2:
		result = zf;
		break;
	case 3:
		result = cf || zf;
		break;
	case 4:
		result = sf;
		break;
	case 5:
		result = pf;
		break;
	case 6:
		result = sf != of;
		break;
	default:
		result = zf || (sf != of);
		break;
	}
	return (cc & 1) ? !result : result;
}
} // namespace

const char *name()
{
	return "x86_64";
}

std::optional<insn_info> decode(const uint8_t *code, std::string &err)
{
	x86_insn insn;
	if (!x86_decode_insn(code, MAX_INSN_LEN, insn)) {
		err = "unable to decode the first instruction of the target";
		return std::nullopt;
	}
	if (insn.len == 1 && code[0] == INT3) {
		err = "the target already starts with an int3 breakpoint";
		return std::nullopt;
	}
	insn_info info;
	info.len = insn.len;
	switch (insn.branch) {
	case x86_branch_kind::none:
		info.kind = insn_kind::execute_out_of_line;
		break;
	case x86_branch_kind::jmp_rel:
	case x86_branch_kind::call_rel:
	case x86_branch_kind::jcc_rel:
		info.kind = insn_kind::emulate;
		break;
	case x86_branch_kind::unsupported:
		err = "the first instruction of the target is a control transfer "
		      "that cannot be relocated (loop/jrcxz/indirect call)";
		return std::nullopt;
	}
	info.riprel = insn.riprel;
	info.disp_off = insn.disp_off;
	return info;
}

size_t trap_bytes(size_t, uint8_t out[MAX_TRAP_LEN])
{
	out[0] = INT3;
	return 1;
}

uintptr_t trap_pc(const ucontext_t *uc)
{
	// int3 is reported with rip pointing after the one byte instruction
	return (uintptr_t)reg(uc, REG_RIP) - 1;
}

void set_pc(ucontext_t *uc, uintptr_t pc)
{
	reg(uc, REG_RIP) = (greg_t)pc;
}

uintptr_t get_sp(const ucontext_t *uc)
{
	return (uintptr_t)reg(uc, REG_RSP);
}

uintptr_t get_return_address(const ucontext_t *uc)
{
	return *(const uintptr_t *)get_sp(uc);
}

void set_return_address(ucontext_t *uc, uintptr_t addr)
{
	*(uintptr_t *)get_sp(uc) = addr;
}

void do_return(ucontext_t *uc, uint64_t value)
{
	uintptr_t sp = get_sp(uc);
	uintptr_t ret = *(uintptr_t *)sp;
	reg(uc, REG_RAX) = (greg_t)value;
	reg(uc, REG_RSP) = (greg_t)(sp + sizeof(uintptr_t));
	reg(uc, REG_RIP) = (greg_t)ret;
}

uint64_t get_return_value(const ucontext_t *uc)
{
	return (uint64_t)reg(uc, REG_RAX);
}

void fill_pt_regs(const ucontext_t *uc, uintptr_t pc, bpftime::pt_regs &regs)
{
	std::memset(&regs, 0, sizeof(regs));
	regs.r15 = reg(uc, REG_R15);
	regs.r14 = reg(uc, REG_R14);
	regs.r13 = reg(uc, REG_R13);
	regs.r12 = reg(uc, REG_R12);
	regs.bp = reg(uc, REG_RBP);
	regs.bx = reg(uc, REG_RBX);
	regs.r11 = reg(uc, REG_R11);
	regs.r10 = reg(uc, REG_R10);
	regs.r9 = reg(uc, REG_R9);
	regs.r8 = reg(uc, REG_R8);
	regs.ax = reg(uc, REG_RAX);
	regs.cx = reg(uc, REG_RCX);
	regs.dx = reg(uc, REG_RDX);
	regs.si = reg(uc, REG_RSI);
	regs.di = reg(uc, REG_RDI);
	regs.orig_ax = (uint64_t)-1;
	regs.ip = pc;
	regs.cs = reg(uc, REG_CSGSFS) & 0xffff;
	regs.flags = reg(uc, REG_EFL);
	regs.sp = reg(uc, REG_RSP);
	regs.ss = (reg(uc, REG_CSGSFS) >> 48) & 0xffff;
}

bool get_arg(const bpftime::pt_regs &regs, unsigned n, uint64_t *value)
{
	switch (n) {
	case 0:
		*value = regs.di;
		return true;
	case 1:
		*value = regs.si;
		return true;
	case 2:
		*value = regs.dx;
		return true;
	case 3:
		*value = regs.cx;
		return true;
	case 4:
		*value = regs.r8;
		return true;
	case 5:
		*value = regs.r9;
		return true;
	default:
		return false;
	}
}

bool prepare_out_of_line(const uint8_t *orig, const insn_info &info,
			 uintptr_t orig_addr, uint8_t *slot, size_t slot_size,
			 size_t *trap_offset, std::string &err)
{
	if ((size_t)info.len + 1 > slot_size) {
		err = "out-of-line slot too small";
		return false;
	}
	std::memcpy(slot, orig, info.len);
	if (info.riprel) {
		// The displacement is relative to the end of the instruction.
		// Executing from the slot changes that base, so shift the
		// displacement by the distance between the two copies.
		int64_t delta = (int64_t)orig_addr - (int64_t)(uintptr_t)slot;
		int32_t disp;
		std::memcpy(&disp, orig + info.disp_off, sizeof(disp));
		int64_t fixed = (int64_t)disp + delta;
		if (fixed != (int64_t)(int32_t)fixed) {
			err = "rip-relative operand cannot reach its target from "
			      "the out-of-line slot";
			return false;
		}
		int32_t fixed32 = (int32_t)fixed;
		std::memcpy(slot + info.disp_off, &fixed32, sizeof(fixed32));
	}
	slot[info.len] = INT3;
	*trap_offset = info.len;
	return true;
}

void emulate(ucontext_t *uc, const uint8_t *orig, const insn_info &info,
	     uintptr_t orig_addr)
{
	x86_insn insn;
	x86_decode_insn(orig, MAX_INSN_LEN, insn);
	uintptr_t next = orig_addr + info.len;
	int64_t rel = 0;
	if (insn.rel_size == 1) {
		rel = (int8_t)orig[insn.rel_off];
	} else {
		int32_t r;
		std::memcpy(&r, orig + insn.rel_off, sizeof(r));
		rel = r;
	}
	uintptr_t target = (uintptr_t)((int64_t)next + rel);
	switch (insn.branch) {
	case x86_branch_kind::jmp_rel:
		set_pc(uc, target);
		break;
	case x86_branch_kind::call_rel: {
		uintptr_t sp = get_sp(uc) - sizeof(uintptr_t);
		*(uintptr_t *)sp = next;
		reg(uc, REG_RSP) = (greg_t)sp;
		set_pc(uc, target);
		break;
	}
	case x86_branch_kind::jcc_rel:
		set_pc(uc, eval_condition(insn.condition, reg(uc, REG_EFL)) ?
				   target :
				   next);
		break;
	default:
		set_pc(uc, next);
		break;
	}
}

void flush_icache(void *, size_t)
{
	// x86 keeps the instruction cache coherent with data writes
}
} // namespace bpftime::attach::trap::arch
#endif
