/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "trap_arch.hpp"
#include <cstring>
#include <sys/ucontext.h>

namespace bpftime::attach::trap::arch
{
namespace
{
constexpr uint32_t EBREAK = 0x00100073;
constexpr uint16_t C_EBREAK = 0x9002;
// Indices into uc_mcontext.__gregs
constexpr unsigned GREG_PC = 0;
constexpr unsigned GREG_RA = 1;
constexpr unsigned GREG_SP = 2;
constexpr unsigned GREG_A0 = 10;

inline uint16_t load16(const uint8_t *p)
{
	uint16_t v;
	std::memcpy(&v, p, sizeof(v));
	return v;
}
inline uint32_t load32(const uint8_t *p)
{
	uint32_t v;
	std::memcpy(&v, p, sizeof(v));
	return v;
}
inline int64_t sext(uint64_t v, unsigned bits)
{
	const unsigned shift = 64 - bits;
	return (int64_t)(v << shift) >> shift;
}
// Integer register x<r>. x0 is hardwired to zero and has no slot in
// __gregs (slot 0 holds the pc), so it is special cased.
inline uint64_t read_reg(const ucontext_t *uc, unsigned r)
{
	return r == 0 ? 0 : (uint64_t)uc->uc_mcontext.__gregs[r];
}
inline void write_reg(ucontext_t *uc, unsigned r, uint64_t v)
{
	if (r != 0)
		uc->uc_mcontext.__gregs[r] = v;
}
inline size_t insn_length(const uint8_t *code)
{
	return (load16(code) & 3) == 3 ? 4 : 2;
}

enum class rv_kind {
	other,
	ebreak,
	// 32-bit
	auipc,
	jal,
	jalr,
	branch,
	// 16-bit
	c_j,
	c_beqz,
	c_bnez,
	c_jr,
	c_jalr,
};

rv_kind classify(const uint8_t *code)
{
	if (insn_length(code) == 4) {
		uint32_t insn = load32(code);
		if (insn == EBREAK)
			return rv_kind::ebreak;
		switch (insn & 0x7f) {
		case 0x17:
			return rv_kind::auipc;
		case 0x6f:
			return rv_kind::jal;
		case 0x67:
			return rv_kind::jalr;
		case 0x63:
			return rv_kind::branch;
		default:
			return rv_kind::other;
		}
	}
	uint16_t insn = load16(code);
	if (insn == C_EBREAK)
		return rv_kind::ebreak;
	unsigned op = insn & 3;
	unsigned funct3 = insn >> 13;
	if (op == 1) {
		if (funct3 == 5)
			return rv_kind::c_j;
		if (funct3 == 6)
			return rv_kind::c_beqz;
		if (funct3 == 7)
			return rv_kind::c_bnez;
	} else if (op == 2 && funct3 == 4) {
		unsigned rs2 = (insn >> 2) & 31;
		unsigned rs1 = (insn >> 7) & 31;
		if (rs2 == 0 && rs1 != 0)
			return (insn & (1u << 12)) ? rv_kind::c_jalr :
						     rv_kind::c_jr;
	}
	return rv_kind::other;
}
} // namespace

const char *name()
{
	return "riscv64";
}

std::optional<insn_info> decode(const uint8_t *code, std::string &err)
{
	insn_info info;
	info.len = (uint8_t)insn_length(code);
	switch (classify(code)) {
	case rv_kind::ebreak:
		err = "the target already starts with an ebreak instruction";
		return std::nullopt;
	case rv_kind::other:
		info.kind = insn_kind::execute_out_of_line;
		break;
	default:
		info.kind = insn_kind::emulate;
		break;
	}
	return info;
}

size_t trap_bytes(size_t insn_len, uint8_t out[MAX_TRAP_LEN])
{
	if (insn_len == 2) {
		std::memcpy(out, &C_EBREAK, 2);
		return 2;
	}
	std::memcpy(out, &EBREAK, 4);
	return 4;
}

uintptr_t trap_pc(const ucontext_t *uc)
{
	return (uintptr_t)uc->uc_mcontext.__gregs[GREG_PC];
}

void set_pc(ucontext_t *uc, uintptr_t pc)
{
	uc->uc_mcontext.__gregs[GREG_PC] = pc;
}

uintptr_t get_sp(const ucontext_t *uc)
{
	return (uintptr_t)uc->uc_mcontext.__gregs[GREG_SP];
}

uintptr_t get_return_address(const ucontext_t *uc)
{
	return (uintptr_t)uc->uc_mcontext.__gregs[GREG_RA];
}

void set_return_address(ucontext_t *uc, uintptr_t addr)
{
	uc->uc_mcontext.__gregs[GREG_RA] = addr;
}

void do_return(ucontext_t *uc, uint64_t value)
{
	uc->uc_mcontext.__gregs[GREG_A0] = value;
	uc->uc_mcontext.__gregs[GREG_PC] = uc->uc_mcontext.__gregs[GREG_RA];
}

uint64_t get_return_value(const ucontext_t *uc)
{
	return (uint64_t)uc->uc_mcontext.__gregs[GREG_A0];
}

void fill_pt_regs(const ucontext_t *uc, uintptr_t pc, bpftime::pt_regs &regs)
{
	static_assert(sizeof(regs) == 32 * sizeof(uint64_t),
		      "pt_regs must mirror __gregs");
	uint64_t *dst = &regs.epc;
	for (unsigned i = 0; i < 32; i++)
		dst[i] = (uint64_t)uc->uc_mcontext.__gregs[i];
	regs.epc = pc;
}

bool get_arg(const bpftime::pt_regs &regs, unsigned n, uint64_t *value)
{
	if (n >= 8)
		return false;
	*value = (&regs.a0)[n];
	return true;
}

bool prepare_out_of_line(const uint8_t *orig, const insn_info &info,
			 uintptr_t, uint8_t *slot, size_t slot_size,
			 size_t *trap_offset, std::string &err)
{
	if ((size_t)info.len + 4 > slot_size) {
		err = "out-of-line slot too small";
		return false;
	}
	std::memcpy(slot, orig, info.len);
	// A 4-byte ebreak may sit at a 2-byte boundary: a compressed first
	// instruction implies the C extension, which allows 16-bit alignment.
	std::memcpy(slot + info.len, &EBREAK, 4);
	*trap_offset = info.len;
	return true;
}

void emulate(ucontext_t *uc, const uint8_t *orig, const insn_info &info,
	     uintptr_t pc)
{
	uintptr_t next = pc + info.len;
	switch (classify(orig)) {
	case rv_kind::auipc: {
		uint32_t insn = load32(orig);
		unsigned rd = (insn >> 7) & 31;
		int64_t imm = (int64_t)(int32_t)(insn & 0xfffff000u);
		write_reg(uc, rd, pc + imm);
		set_pc(uc, next);
		break;
	}
	case rv_kind::jal: {
		uint32_t insn = load32(orig);
		unsigned rd = (insn >> 7) & 31;
		uint64_t raw = (((insn >> 31) & 1) << 20) |
			       (((insn >> 12) & 0xff) << 12) |
			       (((insn >> 20) & 1) << 11) |
			       (((insn >> 21) & 0x3ff) << 1);
		int64_t imm = sext(raw, 21);
		write_reg(uc, rd, next);
		set_pc(uc, pc + imm);
		break;
	}
	case rv_kind::jalr: {
		uint32_t insn = load32(orig);
		unsigned rd = (insn >> 7) & 31;
		unsigned rs1 = (insn >> 15) & 31;
		int64_t imm = (int64_t)((int32_t)insn >> 20);
		uint64_t target = (read_reg(uc, rs1) + imm) & ~(uint64_t)1;
		write_reg(uc, rd, next);
		set_pc(uc, target);
		break;
	}
	case rv_kind::branch: {
		uint32_t insn = load32(orig);
		unsigned funct3 = (insn >> 12) & 7;
		uint64_t a = read_reg(uc, (insn >> 15) & 31);
		uint64_t b = read_reg(uc, (insn >> 20) & 31);
		uint64_t raw = (((insn >> 31) & 1) << 12) |
			       (((insn >> 7) & 1) << 11) |
			       (((insn >> 25) & 0x3f) << 5) |
			       (((insn >> 8) & 0xf) << 1);
		int64_t imm = sext(raw, 13);
		bool taken;
		switch (funct3) {
		case 0:
			taken = a == b;
			break;
		case 1:
			taken = a != b;
			break;
		case 4:
			taken = (int64_t)a < (int64_t)b;
			break;
		case 5:
			taken = (int64_t)a >= (int64_t)b;
			break;
		case 6:
			taken = a < b;
			break;
		case 7:
			taken = a >= b;
			break;
		default:
			taken = false;
			break;
		}
		set_pc(uc, taken ? pc + imm : next);
		break;
	}
	case rv_kind::c_j: {
		uint16_t insn = load16(orig);
		uint64_t raw = (((insn >> 12) & 1) << 11) |
			       (((insn >> 11) & 1) << 4) |
			       (((insn >> 9) & 3) << 8) |
			       (((insn >> 8) & 1) << 10) |
			       (((insn >> 7) & 1) << 6) |
			       (((insn >> 6) & 1) << 7) |
			       (((insn >> 3) & 7) << 1) |
			       (((insn >> 2) & 1) << 5);
		set_pc(uc, pc + sext(raw, 12));
		break;
	}
	case rv_kind::c_beqz:
	case rv_kind::c_bnez: {
		uint16_t insn = load16(orig);
		unsigned rs1 = 8 + ((insn >> 7) & 7);
		uint64_t raw = (((insn >> 12) & 1) << 8) |
			       (((insn >> 10) & 3) << 3) |
			       (((insn >> 5) & 3) << 6) |
			       (((insn >> 3) & 3) << 1) |
			       (((insn >> 2) & 1) << 5);
		int64_t imm = sext(raw, 9);
		uint64_t val = read_reg(uc, rs1);
		bool taken = (classify(orig) == rv_kind::c_beqz) ? val == 0 :
								   val != 0;
		set_pc(uc, taken ? pc + imm : next);
		break;
	}
	case rv_kind::c_jr:
	case rv_kind::c_jalr: {
		uint16_t insn = load16(orig);
		unsigned rs1 = (insn >> 7) & 31;
		uint64_t target = read_reg(uc, rs1) & ~(uint64_t)1;
		if (classify(orig) == rv_kind::c_jalr)
			write_reg(uc, GREG_RA, next);
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
