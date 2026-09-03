/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
// A small 64-bit x86 instruction length decoder. It only needs to answer
// three questions about the first instruction of a function: how long is it,
// does it use a rip-relative memory operand, and is it a relative control
// transfer. Anything it does not understand is reported as undecodable so the
// caller can refuse to place a probe there.
#ifndef _BPFTIME_X86_INSN_DECODE_HPP
#define _BPFTIME_X86_INSN_DECODE_HPP
#include <cstddef>
#include <cstdint>

namespace bpftime::attach::trap::arch
{
enum class x86_branch_kind {
	none,
	jmp_rel,
	call_rel,
	jcc_rel,
	// loop/jrcxz/indirect call: not relocatable by this implementation
	unsupported,
};

struct x86_insn {
	uint8_t len = 0;
	bool riprel = false;
	// Offset of the 32-bit displacement of a rip-relative operand
	uint8_t disp_off = 0;
	x86_branch_kind branch = x86_branch_kind::none;
	// For relative branches: offset and size (1 or 4) of the immediate
	uint8_t rel_off = 0;
	uint8_t rel_size = 0;
	// Condition code of Jcc
	uint8_t condition = 0;
};

// Decode the instruction starting at `code`, reading at most `max_len`
// bytes. Returns false if the bytes do not form a supported instruction.
bool x86_decode_insn(const uint8_t *code, size_t max_len, x86_insn &out);
} // namespace bpftime::attach::trap::arch
#endif
