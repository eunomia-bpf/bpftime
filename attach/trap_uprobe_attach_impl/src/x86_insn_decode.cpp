/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "x86_insn_decode.hpp"

namespace bpftime::attach::trap::arch
{
namespace
{
constexpr size_t X86_MAX_INSN = 15;

bool is_legacy_prefix(uint8_t b)
{
	switch (b) {
	case 0x66:
	case 0x67:
	case 0xf0:
	case 0xf2:
	case 0xf3:
	case 0x2e:
	case 0x36:
	case 0x3e:
	case 0x26:
	case 0x64:
	case 0x65:
		return true;
	default:
		return false;
	}
}

// Opcodes of the 0F map that do not carry a ModRM byte
bool two_byte_has_modrm(uint8_t op)
{
	switch (op) {
	case 0x05: // syscall
	case 0x06: // clts
	case 0x07: // sysret
	case 0x08: // invd
	case 0x09: // wbinvd
	case 0x0b: // ud2
	case 0x0e: // femms
	case 0x30: // wrmsr
	case 0x31: // rdtsc
	case 0x32: // rdmsr
	case 0x33: // rdpmc
	case 0x34: // sysenter
	case 0x35: // sysexit
	case 0x37: // getsec
	case 0x77: // emms
	case 0xa0: // push fs
	case 0xa1: // pop fs
	case 0xa2: // cpuid
	case 0xa8: // push gs
	case 0xa9: // pop gs
	case 0xaa: // rsm
	case 0xc8:
	case 0xc9:
	case 0xca:
	case 0xcb:
	case 0xcc:
	case 0xcd:
	case 0xce:
	case 0xcf: // bswap
		return false;
	default:
		return true;
	}
}

// Opcodes of the 0F map followed by an imm8
bool two_byte_has_imm8(uint8_t op)
{
	switch (op) {
	case 0x70:
	case 0x71:
	case 0x72:
	case 0x73:
	case 0xa4:
	case 0xac:
	case 0xba:
	case 0xc2:
	case 0xc4:
	case 0xc5:
	case 0xc6:
		return true;
	default:
		return false;
	}
}
} // namespace

bool x86_decode_insn(const uint8_t *code, size_t max_len, x86_insn &out)
{
	out = x86_insn{};
	if (max_len > X86_MAX_INSN)
		max_len = X86_MAX_INSN;
	size_t i = 0;
	bool opsize16 = false;
	bool rex_w = false;

	while (i < max_len && is_legacy_prefix(code[i])) {
		if (code[i] == 0x66)
			opsize16 = true;
		i++;
	}
	if (i >= max_len)
		return false;
	if ((code[i] & 0xf0) == 0x40) {
		rex_w = (code[i] & 0x08) != 0;
		i++;
		if (i >= max_len)
			return false;
	}
	uint8_t op = code[i++];
	bool has_modrm = false;
	size_t imm = 0;
	const size_t imm_z = opsize16 ? 2 : 4;

	auto relative_branch = [&](x86_branch_kind kind, size_t rel_size,
				   uint8_t condition) -> bool {
		out.branch = kind;
		out.condition = condition;
		out.rel_off = (uint8_t)i;
		out.rel_size = (uint8_t)rel_size;
		i += rel_size;
		if (i > max_len)
			return false;
		out.len = (uint8_t)i;
		return true;
	};

	if (op == 0xc4 || op == 0xc5) {
		// VEX prefix (always VEX in 64-bit mode)
		int map;
		if (op == 0xc5) {
			if (i >= max_len)
				return false;
			i++;
			map = 1;
		} else {
			if (i + 1 >= max_len)
				return false;
			map = code[i] & 0x1f;
			i += 2;
			if (map < 1 || map > 3)
				return false;
		}
		if (i >= max_len)
			return false;
		op = code[i++];
		// Every VEX encoded instruction has a ModRM except
		// vzeroupper/vzeroall
		has_modrm = !(map == 1 && op == 0x77);
		if (map == 3 || (map == 1 && two_byte_has_imm8(op)))
			imm = 1;
	} else if (op == 0x0f) {
		if (i >= max_len)
			return false;
		op = code[i++];
		if (op == 0x38) {
			if (i >= max_len)
				return false;
			i++;
			has_modrm = true;
		} else if (op == 0x3a) {
			if (i >= max_len)
				return false;
			i++;
			has_modrm = true;
			imm = 1;
		} else if (op == 0x0f) {
			// 3DNow!
			return false;
		} else if (op >= 0x80 && op <= 0x8f) {
			return relative_branch(x86_branch_kind::jcc_rel, 4,
					       op & 0x0f);
		} else {
			has_modrm = two_byte_has_modrm(op);
			if (two_byte_has_imm8(op))
				imm = 1;
		}
	} else if (op < 0x40) {
		// ALU block: add/or/adc/sbb/and/sub/xor/cmp
		switch (op & 7) {
		case 0:
		case 1:
		case 2:
		case 3:
			has_modrm = true;
			break;
		case 4:
			imm = 1;
			break;
		case 5:
			imm = imm_z;
			break;
		default:
			// push/pop segment, daa/aaa/...: invalid in 64-bit
			return false;
		}
	} else if (op >= 0x50 && op <= 0x5f) {
		// push/pop r64
	} else if (op == 0x63) {
		has_modrm = true;
	} else if (op == 0x68) {
		imm = imm_z;
	} else if (op == 0x69) {
		has_modrm = true;
		imm = imm_z;
	} else if (op == 0x6a) {
		imm = 1;
	} else if (op == 0x6b) {
		has_modrm = true;
		imm = 1;
	} else if (op >= 0x6c && op <= 0x6f) {
		// ins/outs
	} else if (op >= 0x70 && op <= 0x7f) {
		return relative_branch(x86_branch_kind::jcc_rel, 1, op & 0x0f);
	} else if (op == 0x80 || op == 0x83) {
		has_modrm = true;
		imm = 1;
	} else if (op == 0x81) {
		has_modrm = true;
		imm = imm_z;
	} else if (op >= 0x84 && op <= 0x8f) {
		has_modrm = true;
	} else if (op >= 0x90 && op <= 0x99) {
		// nop/xchg/cwde/cdq
	} else if (op >= 0x9b && op <= 0x9f) {
		// fwait/pushf/popf/sahf/lahf
	} else if (op >= 0xa0 && op <= 0xa3) {
		// mov moffs64 (64-bit absolute address)
		imm = 8;
	} else if (op >= 0xa4 && op <= 0xa7) {
		// movs/cmps
	} else if (op == 0xa8) {
		imm = 1;
	} else if (op == 0xa9) {
		imm = imm_z;
	} else if (op >= 0xaa && op <= 0xaf) {
		// stos/lods/scas
	} else if (op >= 0xb0 && op <= 0xb7) {
		imm = 1;
	} else if (op >= 0xb8 && op <= 0xbf) {
		imm = rex_w ? 8 : imm_z;
	} else if (op == 0xc0 || op == 0xc1) {
		has_modrm = true;
		imm = 1;
	} else if (op == 0xc2) {
		imm = 2;
	} else if (op == 0xc3) {
		// ret
	} else if (op == 0xc6) {
		has_modrm = true;
		imm = 1;
	} else if (op == 0xc7) {
		has_modrm = true;
		imm = imm_z;
	} else if (op == 0xc8) {
		// enter imm16, imm8
		imm = 3;
	} else if (op == 0xc9 || op == 0xcb || op == 0xcc || op == 0xcf) {
		// leave/retf/int3/iret
	} else if (op == 0xca) {
		imm = 2;
	} else if (op == 0xcd) {
		imm = 1;
	} else if (op >= 0xd0 && op <= 0xd3) {
		has_modrm = true;
	} else if (op == 0xd7) {
		// xlat
	} else if (op >= 0xd8 && op <= 0xdf) {
		// x87
		has_modrm = true;
	} else if (op >= 0xe0 && op <= 0xe3) {
		// loop/loopcc/jrcxz: rel8 but depends on rcx, not emulated
		out.branch = x86_branch_kind::unsupported;
		i += 1;
		if (i > max_len)
			return false;
		out.len = (uint8_t)i;
		return true;
	} else if (op >= 0xe4 && op <= 0xe7) {
		imm = 1;
	} else if (op == 0xe8) {
		return relative_branch(x86_branch_kind::call_rel, 4, 0);
	} else if (op == 0xe9) {
		return relative_branch(x86_branch_kind::jmp_rel, 4, 0);
	} else if (op == 0xeb) {
		return relative_branch(x86_branch_kind::jmp_rel, 1, 0);
	} else if (op >= 0xec && op <= 0xef) {
		// in/out dx
	} else if (op == 0xf1 || op == 0xf4 || op == 0xf5) {
		// int1/hlt/cmc
	} else if (op == 0xf6 || op == 0xf7) {
		has_modrm = true;
		if (i >= max_len)
			return false;
		unsigned reg = (code[i] >> 3) & 7;
		if (reg <= 1)
			imm = (op == 0xf6) ? 1 : imm_z;
	} else if (op >= 0xf8 && op <= 0xfd) {
		// clc/stc/cli/sti/cld/std
	} else if (op == 0xfe) {
		has_modrm = true;
	} else if (op == 0xff) {
		has_modrm = true;
		if (i >= max_len)
			return false;
		unsigned reg = (code[i] >> 3) & 7;
		if (reg == 2 || reg == 3) {
			// indirect call pushes a return address that would
			// point into the out-of-line slot
			out.branch = x86_branch_kind::unsupported;
		}
	} else {
		return false;
	}

	if (has_modrm) {
		if (i >= max_len)
			return false;
		uint8_t modrm = code[i++];
		uint8_t mod = modrm >> 6;
		uint8_t rm = modrm & 7;
		if (mod != 3) {
			if (rm == 4) {
				if (i >= max_len)
					return false;
				uint8_t sib = code[i++];
				if ((sib & 7) == 5 && mod == 0)
					i += 4;
			} else if (rm == 5 && mod == 0) {
				out.riprel = true;
				out.disp_off = (uint8_t)i;
				i += 4;
			}
			if (mod == 1)
				i += 1;
			else if (mod == 2)
				i += 4;
		}
	}
	i += imm;
	if (i > max_len)
		return false;
	out.len = (uint8_t)i;
	return true;
}
} // namespace bpftime::attach::trap::arch
