/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
// Per-architecture primitives used by the trap based uprobe attach
// implementation. Everything the core needs to know about the CPU lives
// behind this interface; the core itself is architecture neutral.
#ifndef _BPFTIME_TRAP_ARCH_HPP
#define _BPFTIME_TRAP_ARCH_HPP

#include "bpftime_pt_regs.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <ucontext.h>

namespace bpftime
{
namespace attach
{
namespace trap
{
namespace arch
{
// Longest instruction we ever copy out of line (x86 is 15 bytes).
constexpr size_t MAX_INSN_LEN = 16;
// Longest trap instruction (aarch64 brk / riscv ebreak are 4 bytes).
constexpr size_t MAX_TRAP_LEN = 4;

enum class insn_kind {
	// The instruction is position independent (after optional fix-ups)
	// and can be executed from the out-of-line slot.
	execute_out_of_line,
	// The instruction is pc-relative and is emulated in software by
	// `emulate()` instead of being executed.
	emulate,
};

struct insn_info {
	// Length in bytes of the first instruction at the probe address.
	uint8_t len = 0;
	insn_kind kind = insn_kind::execute_out_of_line;
	// x86 only: the instruction has a rip-relative memory operand whose
	// 32-bit displacement starts at byte `disp_off` and must be adjusted
	// when the instruction is relocated to the slot.
	bool riprel = false;
	uint8_t disp_off = 0;
};

// Human readable architecture name used in diagnostics.
const char *name();

// Decode the first instruction at `code`. On failure returns std::nullopt and
// stores a reason into `err`.
std::optional<insn_info> decode(const uint8_t *code, std::string &err);

// Fill `out` with the trap instruction used to replace an instruction of
// `insn_len` bytes. Returns the number of bytes written. The trap never
// exceeds the original instruction length.
size_t trap_bytes(size_t insn_len, uint8_t out[MAX_TRAP_LEN]);

// Address of the trap instruction that caused the SIGTRAP (x86 reports the
// address after the int3, this normalizes it).
uintptr_t trap_pc(const ucontext_t *uc);
void set_pc(ucontext_t *uc, uintptr_t pc);
uintptr_t get_sp(const ucontext_t *uc);
// Return address of the function that is being entered
uintptr_t get_return_address(const ucontext_t *uc);
void set_return_address(ucontext_t *uc, uintptr_t addr);
// Make the interrupted function return `value` immediately, as if its body
// had been skipped.
void do_return(ucontext_t *uc, uint64_t value);
uint64_t get_return_value(const ucontext_t *uc);
// Copy the register state into a kernel-style pt_regs. `pc` is the address
// the eBPF program should observe as the instruction pointer.
void fill_pt_regs(const ucontext_t *uc, uintptr_t pc, bpftime::pt_regs &regs);
// n-th (0 based) integer argument according to the calling convention.
// Returns false when the argument index is not available in registers.
bool get_arg(const bpftime::pt_regs &regs, unsigned n, uint64_t *value);

// Write the relocated instruction plus a trap into `slot` (of `slot_size`
// bytes). `trap_offset` receives the offset of the trap within the slot.
bool prepare_out_of_line(const uint8_t *orig, const insn_info &info,
			 uintptr_t orig_addr, uint8_t *slot, size_t slot_size,
			 size_t *trap_offset, std::string &err);

// Emulate an `insn_kind::emulate` instruction and advance the pc.
void emulate(ucontext_t *uc, const uint8_t *orig, const insn_info &info,
	     uintptr_t orig_addr);

void flush_icache(void *addr, size_t len);
} // namespace arch
} // namespace trap
} // namespace attach
} // namespace bpftime

#endif
