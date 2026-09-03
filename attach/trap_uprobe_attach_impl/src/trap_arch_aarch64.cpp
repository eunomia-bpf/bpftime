/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#if defined(__aarch64__) || defined(_M_ARM64)
#include "trap_arch.hpp"
#include <cstring>
#include <sys/ucontext.h>

namespace bpftime::attach::trap::arch
{
namespace
{
constexpr uint32_t BRK0 = 0xd4200000; // brk #0

inline uint32_t load32(const uint8_t *p)
{
	uint32_t v;
	std::memcpy(&v, p, sizeof(v));
	return v;
}
inline void store32(uint8_t *p, uint32_t v)
{
	std::memcpy(p, &v, sizeof(v));
}
inline int64_t sext(uint64_t v, unsigned bits)
{
	const unsigned shift = 64 - bits;
	return (int64_t)(v << shift) >> shift;
}
inline uint64_t read_reg(const ucontext_t *uc, unsigned r)
{
	// Register 31 is the zero register in the encodings we emulate
	return r < 31 ? uc->uc_mcontext.regs[r] : 0;
}
inline void write_reg(ucontext_t *uc, unsigned r, uint64_t v)
{
	if (r < 31)
		uc->uc_mcontext.regs[r] = v;
}

enum class a64_kind {
	other,
	brk,
	adr,
	b,
	b_cond,
	cbz,
	tbz,
	ldr_literal,
	blr,
};

a64_kind classify(uint32_t insn)
{
	if ((insn & 0xffe0001f) == BRK0)
		return a64_kind::brk;
	if ((insn & 0x1f000000) == 0x10000000)
		return a64_kind::adr;
	if ((insn & 0x7c000000) == 0x14000000)
		return a64_kind::b;
	if ((insn & 0xff000010) == 0x54000000)
		return a64_kind::b_cond;
	if ((insn & 0x7e000000) == 0x34000000)
		return a64_kind::cbz;
	if ((insn & 0x7e000000) == 0x36000000)
		return a64_kind::tbz;
	if ((insn & 0x3b000000) == 0x18000000)
		return a64_kind::ldr_literal;
	if ((insn & 0xfffffc1f) == 0xd63f0000)
		return a64_kind::blr;
	return a64_kind::other;
}

bool eval_condition(unsigned cond, uint64_t pstate)
{
	const bool n = pstate & (1u << 31);
	const bool z = pstate & (1u << 30);
	const bool c = pstate & (1u << 29);
	const bool v = pstate & (1u << 28);
	bool result;
	switch (cond >> 1) {
	case 0:
		result = z;
		break;
	case 1:
		result = c;
		break;
	case 2:
		result = n;
		break;
	case 3:
		result = v;
		break;
	case 4:
		result = c && !z;
		break;
	case 5:
		result = n == v;
		break;
	case 6:
		result = !z && (n == v);
		break;
	default:
		// AL / NV both mean always
		return true;
	}
	return (cond & 1) ? !result : result;
}
} // namespace

const char *name()
{
	return "aarch64";
}

std::optional<insn_info> decode(const uint8_t *code, std::string &err)
{
	uint32_t insn = load32(code);
	insn_info info;
	info.len = 4;
	switch (classify(insn)) {
	case a64_kind::brk:
		err = "the target already starts with a brk instruction";
		return std::nullopt;
	case a64_kind::other:
		info.kind = insn_kind::execute_out_of_line;
		break;
	default:
		info.kind = insn_kind::emulate;
		break;
	}
	return info;
}

size_t trap_bytes(size_t, uint8_t out[MAX_TRAP_LEN])
{
	store32(out, BRK0);
	return 4;
}

uintptr_t trap_pc(const ucontext_t *uc)
{
	return (uintptr_t)uc->uc_mcontext.pc;
}

void set_pc(ucontext_t *uc, uintptr_t pc)
{
	uc->uc_mcontext.pc = pc;
}

uintptr_t get_sp(const ucontext_t *uc)
{
	return (uintptr_t)uc->uc_mcontext.sp;
}

uintptr_t get_return_address(const ucontext_t *uc)
{
	return (uintptr_t)uc->uc_mcontext.regs[30];
}

void set_return_address(ucontext_t *uc, uintptr_t addr)
{
	uc->uc_mcontext.regs[30] = addr;
}

void do_return(ucontext_t *uc, uint64_t value)
{
	uc->uc_mcontext.regs[0] = value;
	uc->uc_mcontext.pc = uc->uc_mcontext.regs[30];
}

uint64_t get_return_value(const ucontext_t *uc)
{
	return uc->uc_mcontext.regs[0];
}

void fill_pt_regs(const ucontext_t *uc, uintptr_t pc, bpftime::pt_regs &regs)
{
	for (unsigned i = 0; i < 31; i++)
		regs.regs[i] = uc->uc_mcontext.regs[i];
	regs.sp = uc->uc_mcontext.sp;
	regs.pc = pc;
	regs.pstate = uc->uc_mcontext.pstate;
}

bool get_arg(const bpftime::pt_regs &regs, unsigned n, uint64_t *value)
{
	if (n >= 8)
		return false;
	*value = regs.regs[n];
	return true;
}

bool prepare_out_of_line(const uint8_t *orig, const insn_info &info,
			 uintptr_t, uint8_t *slot, size_t slot_size,
			 size_t *trap_offset, std::string &err)
{
	if (slot_size < 8) {
		err = "out-of-line slot too small";
		return false;
	}
	std::memcpy(slot, orig, 4);
	store32(slot + 4, BRK0);
	*trap_offset = 4;
	(void)info;
	return true;
}

void emulate(ucontext_t *uc, const uint8_t *orig, const insn_info &,
	     uintptr_t pc)
{
	uint32_t insn = load32(orig);
	uintptr_t next = pc + 4;
	switch (classify(insn)) {
	case a64_kind::adr: {
		unsigned rd = insn & 31;
		uint64_t immlo = (insn >> 29) & 3;
		uint64_t immhi = (insn >> 5) & 0x7ffff;
		int64_t imm = sext((immhi << 2) | immlo, 21);
		if (insn & (1u << 31))
			write_reg(uc, rd, (pc & ~(uint64_t)0xfff) + (imm << 12));
		else
			write_reg(uc, rd, pc + imm);
		set_pc(uc, next);
		break;
	}
	case a64_kind::b: {
		int64_t imm = sext(insn & 0x3ffffff, 26) << 2;
		if (insn & (1u << 31))
			write_reg(uc, 30, next);
		set_pc(uc, pc + imm);
		break;
	}
	case a64_kind::b_cond: {
		int64_t imm = sext((insn >> 5) & 0x7ffff, 19) << 2;
		bool taken = eval_condition(insn & 0xf, uc->uc_mcontext.pstate);
		set_pc(uc, taken ? pc + imm : next);
		break;
	}
	case a64_kind::cbz: {
		int64_t imm = sext((insn >> 5) & 0x7ffff, 19) << 2;
		uint64_t val = read_reg(uc, insn & 31);
		if (!(insn & (1u << 31)))
			val &= 0xffffffffu;
		bool nonzero_form = insn & (1u << 24);
		bool taken = nonzero_form ? val != 0 : val == 0;
		set_pc(uc, taken ? pc + imm : next);
		break;
	}
	case a64_kind::tbz: {
		unsigned bit = ((insn >> 31) << 5) | ((insn >> 19) & 0x1f);
		int64_t imm = sext((insn >> 5) & 0x3fff, 14) << 2;
		uint64_t val = read_reg(uc, insn & 31);
		bool set = (val >> bit) & 1;
		bool nonzero_form = insn & (1u << 24);
		bool taken = nonzero_form ? set : !set;
		set_pc(uc, taken ? pc + imm : next);
		break;
	}
	case a64_kind::ldr_literal: {
		int64_t imm = sext((insn >> 5) & 0x7ffff, 19) << 2;
		uintptr_t addr = pc + imm;
		unsigned rt = insn & 31;
		switch (insn >> 30) {
		case 0: {
			uint32_t v;
			std::memcpy(&v, (void *)addr, sizeof(v));
			write_reg(uc, rt, v);
			break;
		}
		case 1: {
			uint64_t v;
			std::memcpy(&v, (void *)addr, sizeof(v));
			write_reg(uc, rt, v);
			break;
		}
		case 2: {
			int32_t v;
			std::memcpy(&v, (void *)addr, sizeof(v));
			write_reg(uc, rt, (uint64_t)(int64_t)v);
			break;
		}
		default:
			// prfm: hint only
			break;
		}
		set_pc(uc, next);
		break;
	}
	case a64_kind::blr: {
		uint64_t target = read_reg(uc, (insn >> 5) & 31);
		write_reg(uc, 30, next);
		set_pc(uc, target);
		break;
	}
	default:
		set_pc(uc, next);
		break;
	}
}

void flush_icache(void *addr, size_t len)
{
	__builtin___clear_cache((char *)addr, (char *)addr + len);
}
} // namespace bpftime::attach::trap::arch
#endif
