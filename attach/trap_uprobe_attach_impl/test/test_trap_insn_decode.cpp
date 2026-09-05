#include "trap_test_common.hpp"
#include <trap_uprobe_attach_impl.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>

using namespace bpftime;
using namespace bpftime::attach::trap;

extern "C" uint64_t g_trap_global;
extern "C" uint64_t *__trap_rv_auipc_first();
extern "C" uint64_t __trap_rv_jal_first(uint64_t);
extern "C" uint64_t __trap_rv_beqz_first(uint64_t);
extern "C" uint64_t __trap_rv_bge_first(uint64_t, uint64_t);
extern "C" uint64_t __trap_rv_cj_first(uint64_t);
extern "C" uint64_t __trap_rv_cbnez_first(uint64_t);
extern "C" uint64_t __trap_rv_cjr_first(uint64_t, uint64_t (*)(uint64_t));
extern "C" uint64_t __trap_rv_jalr_first(uint64_t, uint64_t (*)(uint64_t));
extern "C" void __trap_rv_ebreak_first();
extern "C" void __trap_rv_cebreak_first();

extern "C" uint64_t __trap_rv_misaligned4(uint64_t);

extern "C" TRAP_TEST_TARGET uint64_t __trap_call_helper(uint64_t a)
{
	asm("");
	return a + 1000;
}

// A 4-byte instruction whose address is ≡ 2 (mod 4). This is the exact
// precondition that triggers the three-phase write (c.ebreak staging)
// in write_code.  The c.nop before the beq shifts it to a 2-mod-4 addr.
asm(".option push\n"
    ".option rvc\n"
    ".balign 4\n"
    ".globl __trap_rv_misaligned4\n"
    ".type __trap_rv_misaligned4, @function\n"
    "__trap_rv_misaligned4:\n"
    "	c.nop\n"		// 2 bytes: pushes the next insn to addr+2
    "	beq a0, x0, 1f\n"	// 4 bytes at addr ≡ 2 (mod 4) — three-phase target
    "	li a0, 1\n"
    "	ret\n"
    "1:	li a0, 2\n"
    "	ret\n"
    ".size __trap_rv_misaligned4, .-__trap_rv_misaligned4\n"
    ".option pop\n");

asm(".data\n"
    ".balign 8\n"
    ".globl g_trap_global\n"
    ".type g_trap_global, @object\n"
    "g_trap_global:\n"
    "	.quad 0x1234\n"
    ".size g_trap_global, 8\n"
    ".text\n"
    ".globl __trap_rv_auipc_first\n"
    ".type __trap_rv_auipc_first, @function\n"
    "__trap_rv_auipc_first:\n"
    "1:	auipc a0, %pcrel_hi(g_trap_global)\n"
    "	addi a0, a0, %pcrel_lo(1b)\n"
    "	ret\n"
    ".size __trap_rv_auipc_first, .-__trap_rv_auipc_first\n"
    ".option push\n"
    ".option norvc\n"
    ".globl __trap_rv_jal_first\n"
    ".type __trap_rv_jal_first, @function\n"
    "__trap_rv_jal_first:\n"
    "	jal x0, 1f\n"
    "	li a0, 0\n"
    "	ret\n"
    "1:	addi a0, a0, 42\n"
    "	ret\n"
    ".size __trap_rv_jal_first, .-__trap_rv_jal_first\n"
    ".globl __trap_rv_beqz_first\n"
    ".type __trap_rv_beqz_first, @function\n"
    "__trap_rv_beqz_first:\n"
    "	beq a0, x0, 1f\n"
    "	li a0, 1\n"
    "	ret\n"
    "1:	li a0, 2\n"
    "	ret\n"
    ".size __trap_rv_beqz_first, .-__trap_rv_beqz_first\n"
    ".globl __trap_rv_bge_first\n"
    ".type __trap_rv_bge_first, @function\n"
    "__trap_rv_bge_first:\n"
    "	bge a0, a1, 1f\n"
    "	li a0, 1\n"
    "	ret\n"
    "1:	li a0, 2\n"
    "	ret\n"
    ".size __trap_rv_bge_first, .-__trap_rv_bge_first\n"
    ".globl __trap_rv_jalr_first\n"
    ".type __trap_rv_jalr_first, @function\n"
    "__trap_rv_jalr_first:\n"
    "	jalr x0, 0(a1)\n"
    ".size __trap_rv_jalr_first, .-__trap_rv_jalr_first\n"
    ".option pop\n"
    ".option push\n"
    ".option rvc\n"
    ".globl __trap_rv_cj_first\n"
    ".type __trap_rv_cj_first, @function\n"
    "__trap_rv_cj_first:\n"
    "	c.j 1f\n"
    "	li a0, 0\n"
    "	ret\n"
    "1:	addi a0, a0, 7\n"
    "	ret\n"
    ".size __trap_rv_cj_first, .-__trap_rv_cj_first\n"
    ".globl __trap_rv_cbnez_first\n"
    ".type __trap_rv_cbnez_first, @function\n"
    "__trap_rv_cbnez_first:\n"
    "	c.bnez a0, 1f\n"
    "	li a0, 1\n"
    "	ret\n"
    "1:	li a0, 2\n"
    "	ret\n"
    ".size __trap_rv_cbnez_first, .-__trap_rv_cbnez_first\n"
    ".globl __trap_rv_cjr_first\n"
    ".type __trap_rv_cjr_first, @function\n"
    "__trap_rv_cjr_first:\n"
    "	c.jr a1\n"
    ".size __trap_rv_cjr_first, .-__trap_rv_cjr_first\n"
    ".globl __trap_rv_cebreak_first\n"
    ".type __trap_rv_cebreak_first, @function\n"
    "__trap_rv_cebreak_first:\n"
    "	c.ebreak\n"
    "	ret\n"
    ".size __trap_rv_cebreak_first, .-__trap_rv_cebreak_first\n"
    ".option pop\n"
    ".globl __trap_rv_ebreak_first\n"
    ".type __trap_rv_ebreak_first, @function\n"
    "__trap_rv_ebreak_first:\n"
    "	ebreak\n"
    "	ret\n"
    ".size __trap_rv_ebreak_first, .-__trap_rv_ebreak_first\n");

TEST_CASE("Trap backend (riscv64): emulated first instructions")
{
	trap_attach_impl man;
	int hits = 0;
	auto cb = [&](const pt_regs &) { hits++; };
	REQUIRE(man.create_uprobe_at((void *)&__trap_rv_auipc_first, cb) >= 0);
	REQUIRE(__trap_rv_auipc_first() == &g_trap_global);
	REQUIRE(hits == 1);

	REQUIRE(man.create_uprobe_at((void *)&__trap_rv_jal_first, cb) >= 0);
	REQUIRE(__trap_rv_jal_first(8) == 50);
	REQUIRE(hits == 2);

	REQUIRE(man.create_uprobe_at((void *)&__trap_rv_beqz_first, cb) >= 0);
	REQUIRE(__trap_rv_beqz_first(0) == 2);
	REQUIRE(__trap_rv_beqz_first(9) == 1);
	REQUIRE(hits == 4);

	REQUIRE(man.create_uprobe_at((void *)&__trap_rv_bge_first, cb) >= 0);
	REQUIRE(__trap_rv_bge_first(5, 5) == 2);
	REQUIRE(__trap_rv_bge_first(4, 5) == 1);
	REQUIRE(__trap_rv_bge_first((uint64_t)-1, 5) == 1); // signed compare
	REQUIRE(hits == 7);

	REQUIRE(man.create_uprobe_at((void *)&__trap_rv_cj_first, cb) >= 0);
	REQUIRE(__trap_rv_cj_first(1) == 8);
	REQUIRE(hits == 8);

	REQUIRE(man.create_uprobe_at((void *)&__trap_rv_cbnez_first, cb) >= 0);
	REQUIRE(__trap_rv_cbnez_first(0) == 1);
	REQUIRE(__trap_rv_cbnez_first(3) == 2);
	REQUIRE(hits == 10);

	// Tail calls through a register: the uretprobe must still fire when
	// the callee returns to our hijacked return address
	REQUIRE(man.create_uprobe_at((void *)&__trap_rv_cjr_first, cb) >= 0);
	REQUIRE(man.create_uretprobe_at((void *)&__trap_rv_cjr_first,
					[&](const pt_regs &regs) {
						REQUIRE(PT_REGS_RC(&regs) ==
							1005);
						hits++;
					}) >= 0);
	REQUIRE(__trap_rv_cjr_first(5, __trap_call_helper) == 1005);
	REQUIRE(hits == 12);

	REQUIRE(man.create_uprobe_at((void *)&__trap_rv_jalr_first, cb) >= 0);
	REQUIRE(__trap_rv_jalr_first(6, __trap_call_helper) == 1006);
	REQUIRE(hits == 13);
}

TEST_CASE("Trap backend (riscv64): refuses existing breakpoints")
{
	REQUIRE(check_probe_target((void *)&__trap_rv_ebreak_first).has_value());
	REQUIRE(check_probe_target((void *)&__trap_rv_cebreak_first)
			.has_value());
	REQUIRE(check_probe_target(nullptr).has_value());
	REQUIRE(check_probe_target((const void *)&__trap_call_helper) ==
		std::nullopt);
}

TEST_CASE("Trap backend (riscv64): three-phase patch on 4-byte insn at "
	   "addr % 4 == 2")
{
	// The function starts with c.nop (2 bytes), so the first 4-byte
	// instruction (beq) sits at function_addr + 2, which is ≡ 2 (mod 4).
	// Probing the function puts c.ebreak at function_addr, which is the
	// c.nop — the beq itself is not the probe target, but the function
	// entry exercises the three-phase path when restoring/arming.
	//
	// We probe at the function entry: write_code must patch 4 bytes at a
	// 4-aligned address covering both the c.nop and the first half of beq,
	// OR patch the c.nop alone (2 bytes). Either way the three-phase
	// protocol is involved because the site straddles a 4-byte boundary.
	auto *fn = (uint8_t *)__trap_rv_misaligned4;
	uintptr_t beq_addr = (uintptr_t)fn + 2;
	INFO("fn=" << (void *)fn << " beq at " << (void *)beq_addr);
	REQUIRE((beq_addr % 4) == 2);

	trap_attach_impl man;
	int hits = 0;
	auto cb = [&](const pt_regs &) { hits++; };
	REQUIRE(man.create_uprobe_at((void *)fn, cb) >= 0);
	REQUIRE(__trap_rv_misaligned4(0) == 2);
	REQUIRE(hits == 1);
	REQUIRE(__trap_rv_misaligned4(5) == 1);
	REQUIRE(hits == 2);
}
