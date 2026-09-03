#include "trap_test_common.hpp"
#include <trap_uprobe_attach_impl.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>

using namespace bpftime;
using namespace bpftime::attach::trap;

#if defined(__x86_64__)
#include "x86_insn_decode.hpp"
using namespace bpftime::attach::trap::arch;

static x86_insn decode(std::initializer_list<uint8_t> bytes)
{
	uint8_t buf[16] = {};
	size_t i = 0;
	for (auto b : bytes)
		buf[i++] = b;
	x86_insn insn;
	REQUIRE(x86_decode_insn(buf, sizeof(buf), insn));
	return insn;
}

TEST_CASE("x86 decoder: common prologue instructions")
{
	REQUIRE(decode({ 0xf3, 0x0f, 0x1e, 0xfa }).len == 4); // endbr64
	REQUIRE(decode({ 0x55 }).len == 1); // push rbp
	REQUIRE(decode({ 0x48, 0x89, 0xe5 }).len == 3); // mov rbp,rsp
	REQUIRE(decode({ 0x48, 0x83, 0xec, 0x28 }).len == 4); // sub rsp,0x28
	REQUIRE(decode({ 0x48, 0x81, 0xec, 0x00, 0x01, 0x00, 0x00 }).len ==
		7); // sub rsp,0x100
	REQUIRE(decode({ 0x41, 0x57 }).len == 2); // push r15
	REQUIRE(decode({ 0x48, 0x89, 0x7c, 0x24, 0x08 }).len ==
		5); // mov [rsp+8],rdi
	REQUIRE(decode({ 0x31, 0xc0 }).len == 2); // xor eax,eax
	REQUIRE(decode({ 0xc3 }).len == 1); // ret
	REQUIRE(decode({ 0x0f, 0x1f, 0x44, 0x00, 0x00 }).len == 5); // nopl
	REQUIRE(decode({ 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00 }).len == 6);
	REQUIRE(decode({ 0x48, 0xb8, 1, 2, 3, 4, 5, 6, 7, 8 }).len ==
		10); // movabs rax,imm64
	REQUIRE(decode({ 0xb8, 1, 2, 3, 4 }).len == 5); // mov eax,imm32
	REQUIRE(decode({ 0x66, 0xb8, 1, 2 }).len == 4); // mov ax,imm16
	REQUIRE(decode({ 0xc5, 0xf8, 0x77 }).len == 3); // vzeroupper
	REQUIRE(decode({ 0xc5, 0xfd, 0x6f, 0x07 }).len ==
		4); // vmovdqa ymm0,[rdi]
	REQUIRE(decode({ 0xc4, 0xe2, 0x79, 0x18, 0x07 }).len ==
		5); // vbroadcastss xmm0,[rdi]
	REQUIRE(decode({ 0x64, 0x48, 0x8b, 0x04, 0x25, 0x28, 0, 0, 0 }).len ==
		9); // mov rax,fs:[0x28]
	REQUIRE(decode({ 0xf6, 0x07, 0x01 }).len == 3); // test byte [rdi],1
	REQUIRE(decode({ 0xf7, 0xc7, 1, 2, 3, 4 }).len == 6); // test edi,imm32
	REQUIRE(decode({ 0xf7, 0xd8 }).len == 2); // neg eax
	REQUIRE(decode({ 0x0f, 0x05 }).len == 2); // syscall
	REQUIRE(decode({ 0x0f, 0x0b }).len == 2); // ud2
}

TEST_CASE("x86 decoder: rip-relative operands")
{
	auto lea = decode({ 0x48, 0x8d, 0x05, 0x10, 0x20, 0x30, 0x40 });
	REQUIRE(lea.len == 7);
	REQUIRE(lea.riprel);
	REQUIRE(lea.disp_off == 3);
	auto cmp = decode({ 0x83, 0x3d, 1, 2, 3, 4, 0x05 }); // cmp [rip+x],5
	REQUIRE(cmp.len == 7);
	REQUIRE(cmp.riprel);
	REQUIRE(cmp.disp_off == 2);
	auto jmp_ind = decode({ 0xff, 0x25, 1, 2, 3, 4 }); // jmp [rip+x]
	REQUIRE(jmp_ind.len == 6);
	REQUIRE(jmp_ind.riprel);
	REQUIRE(jmp_ind.branch == x86_branch_kind::none);
	// [rsp] uses a SIB byte, not rip
	REQUIRE(!decode({ 0x48, 0x8b, 0x04, 0x24 }).riprel);
	// [rbp+disp8] with mod=1 is not rip-relative
	auto rbp = decode({ 0x48, 0x8b, 0x45, 0xf8 });
	REQUIRE(rbp.len == 4);
	REQUIRE(!rbp.riprel);
}

TEST_CASE("x86 decoder: control transfers")
{
	auto jmp8 = decode({ 0xeb, 0x10 });
	REQUIRE(jmp8.branch == x86_branch_kind::jmp_rel);
	REQUIRE(jmp8.len == 2);
	REQUIRE(jmp8.rel_size == 1);
	REQUIRE(jmp8.rel_off == 1);
	auto jmp32 = decode({ 0xe9, 1, 2, 3, 4 });
	REQUIRE(jmp32.branch == x86_branch_kind::jmp_rel);
	REQUIRE(jmp32.len == 5);
	REQUIRE(jmp32.rel_size == 4);
	auto call = decode({ 0xe8, 1, 2, 3, 4 });
	REQUIRE(call.branch == x86_branch_kind::call_rel);
	REQUIRE(call.len == 5);
	auto jz = decode({ 0x74, 0x05 });
	REQUIRE(jz.branch == x86_branch_kind::jcc_rel);
	REQUIRE(jz.condition == 4);
	auto jne32 = decode({ 0x0f, 0x85, 1, 2, 3, 4 });
	REQUIRE(jne32.branch == x86_branch_kind::jcc_rel);
	REQUIRE(jne32.len == 6);
	REQUIRE(jne32.condition == 5);
	REQUIRE(decode({ 0xff, 0x15, 1, 2, 3, 4 }).branch ==
		x86_branch_kind::unsupported); // call [rip+x]
	REQUIRE(decode({ 0xff, 0xd0 }).branch ==
		x86_branch_kind::unsupported); // call rax
	REQUIRE(decode({ 0xe2, 0x10 }).branch ==
		x86_branch_kind::unsupported); // loop
	x86_insn bad;
	uint8_t invalid[16] = { 0x06 };
	REQUIRE(!x86_decode_insn(invalid, sizeof(invalid), bad));
	uint8_t truncated[2] = { 0x48, 0xb8 };
	REQUIRE(!x86_decode_insn(truncated, sizeof(truncated), bad));
}

// Real functions whose first instruction needs relocation work. They are
// written in file scope assembly so that no endbr64 is inserted in front of
// the instruction under test.
// Defined in the assembly block below (LTO would otherwise discard a
// C++ definition that is only referenced from inline asm).
extern "C" uint64_t g_trap_global;
extern "C" uint64_t *__trap_riprel_first();
extern "C" uint64_t __trap_jmp_first(uint64_t);
extern "C" uint64_t __trap_call_first(uint64_t);
extern "C" void __trap_indirect_call_first();

extern "C" TRAP_TEST_TARGET uint64_t __trap_call_helper(uint64_t a)
{
	asm("");
	return a + 1000;
}

asm(".data\n"
    ".balign 8\n"
    ".globl g_trap_global\n"
    ".type g_trap_global, @object\n"
    "g_trap_global:\n"
    "	.quad 0x1234\n"
    ".size g_trap_global, 8\n"
    ".text\n"
    ".globl __trap_riprel_first\n"
    ".type __trap_riprel_first, @function\n"
    "__trap_riprel_first:\n"
    "	lea g_trap_global(%rip), %rax\n"
    "	ret\n"
    ".size __trap_riprel_first, .-__trap_riprel_first\n"
    ".globl __trap_jmp_first\n"
    ".type __trap_jmp_first, @function\n"
    "__trap_jmp_first:\n"
    "	jmp 1f\n"
    "	mov $0, %eax\n"
    "	ret\n"
    "1:\n"
    "	lea 42(%rdi), %rax\n"
    "	ret\n"
    ".size __trap_jmp_first, .-__trap_jmp_first\n"
    ".globl __trap_call_first\n"
    ".type __trap_call_first, @function\n"
    "__trap_call_first:\n"
    "	call __trap_call_helper\n"
    "	ret\n"
    ".size __trap_call_first, .-__trap_call_first\n"
    ".globl __trap_indirect_call_first\n"
    ".type __trap_indirect_call_first, @function\n"
    "__trap_indirect_call_first:\n"
    "	call *%rax\n"
    "	ret\n"
    ".size __trap_indirect_call_first, .-__trap_indirect_call_first\n");

TEST_CASE("Trap backend: probes on rip-relative and branch first instructions")
{
	trap_attach_impl man;
	int hits = 0;
	auto cb = [&](const pt_regs &) { hits++; };
	REQUIRE(check_probe_target((void *)&__trap_riprel_first) ==
		std::nullopt);
	REQUIRE(man.create_uprobe_at((void *)&__trap_riprel_first, cb) >= 0);
	REQUIRE(__trap_riprel_first() == &g_trap_global);
	REQUIRE(*__trap_riprel_first() == 0x1234);
	REQUIRE(hits == 2);

	REQUIRE(man.create_uprobe_at((void *)&__trap_jmp_first, cb) >= 0);
	REQUIRE(__trap_jmp_first(8) == 50);
	REQUIRE(hits == 3);

	REQUIRE(man.create_uprobe_at((void *)&__trap_call_first, cb) >= 0);
	REQUIRE(man.create_uretprobe_at((void *)&__trap_call_first,
					[&](const pt_regs &regs) {
						REQUIRE(PT_REGS_RC(&regs) ==
							1005);
						hits++;
					}) >= 0);
	REQUIRE(__trap_call_first(5) == 1005);
	REQUIRE(hits == 5);
}

TEST_CASE("Trap backend: refuses instructions it cannot relocate")
{
	trap_attach_impl man;
	REQUIRE(check_probe_target((void *)&__trap_indirect_call_first)
			.has_value());
	REQUIRE(man.create_uprobe_at((void *)&__trap_indirect_call_first,
				     [](const pt_regs &) {}) == -ENOTSUP);
	REQUIRE(check_probe_target(nullptr).has_value());
}
#elif defined(__riscv) && __riscv_xlen == 64
// Functions whose first instruction is pc-relative or a jump, so the probe
// must emulate it instead of executing it out of line.
// Defined in the assembly block below (LTO would otherwise discard a
// C++ definition that is only referenced from inline asm).
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

extern "C" TRAP_TEST_TARGET uint64_t __trap_call_helper(uint64_t a)
{
	asm("");
	return a + 1000;
}

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
#elif defined(__aarch64__)
// Defined in the assembly block below (LTO would otherwise discard a
// C++ definition that is only referenced from inline asm).
extern "C" uint64_t g_trap_global;
extern "C" uint64_t *__trap_a64_adrp_first();
extern "C" uint64_t __trap_a64_b_first(uint64_t);
extern "C" uint64_t __trap_a64_cbz_first(uint64_t);
extern "C" uint64_t __trap_a64_tbnz_first(uint64_t);
extern "C" uint64_t __trap_a64_ldr_lit_first();
extern "C" uint64_t __trap_a64_tailcall_first(uint64_t);
extern "C" uint64_t __trap_a64_br_first(uint64_t, uint64_t (*)(uint64_t));
extern "C" void __trap_a64_brk_first();

extern "C" TRAP_TEST_TARGET uint64_t __trap_call_helper(uint64_t a)
{
	asm("");
	return a + 1000;
}

asm(".data\n"
    ".balign 8\n"
    ".globl g_trap_global\n"
    ".type g_trap_global, %object\n"
    "g_trap_global:\n"
    "	.quad 0x1234\n"
    ".size g_trap_global, 8\n"
    ".text\n"
    ".globl __trap_a64_adrp_first\n"
    ".type __trap_a64_adrp_first, %function\n"
    "__trap_a64_adrp_first:\n"
    "	adrp x0, g_trap_global\n"
    "	add x0, x0, :lo12:g_trap_global\n"
    "	ret\n"
    ".size __trap_a64_adrp_first, .-__trap_a64_adrp_first\n"
    ".globl __trap_a64_b_first\n"
    ".type __trap_a64_b_first, %function\n"
    "__trap_a64_b_first:\n"
    "	b 1f\n"
    "	mov x0, #0\n"
    "	ret\n"
    "1:	add x0, x0, #42\n"
    "	ret\n"
    ".size __trap_a64_b_first, .-__trap_a64_b_first\n"
    ".globl __trap_a64_cbz_first\n"
    ".type __trap_a64_cbz_first, %function\n"
    "__trap_a64_cbz_first:\n"
    "	cbz x0, 1f\n"
    "	mov x0, #1\n"
    "	ret\n"
    "1:	mov x0, #2\n"
    "	ret\n"
    ".size __trap_a64_cbz_first, .-__trap_a64_cbz_first\n"
    ".globl __trap_a64_tbnz_first\n"
    ".type __trap_a64_tbnz_first, %function\n"
    "__trap_a64_tbnz_first:\n"
    "	tbnz x0, #0, 1f\n"
    "	mov x0, #1\n"
    "	ret\n"
    "1:	mov x0, #2\n"
    "	ret\n"
    ".size __trap_a64_tbnz_first, .-__trap_a64_tbnz_first\n"
    ".globl __trap_a64_ldr_lit_first\n"
    ".type __trap_a64_ldr_lit_first, %function\n"
    "__trap_a64_ldr_lit_first:\n"
    "	ldr x0, 1f\n"
    "	ret\n"
    ".balign 8\n"
    "1:	.quad 0x1122334455667788\n"
    ".size __trap_a64_ldr_lit_first, .-__trap_a64_ldr_lit_first\n"
    ".globl __trap_a64_tailcall_first\n"
    ".type __trap_a64_tailcall_first, %function\n"
    "__trap_a64_tailcall_first:\n"
    "	b __trap_call_helper\n"
    ".size __trap_a64_tailcall_first, .-__trap_a64_tailcall_first\n"
    ".globl __trap_a64_br_first\n"
    ".type __trap_a64_br_first, %function\n"
    "__trap_a64_br_first:\n"
    "	br x1\n"
    ".size __trap_a64_br_first, .-__trap_a64_br_first\n"
    ".globl __trap_a64_brk_first\n"
    ".type __trap_a64_brk_first, %function\n"
    "__trap_a64_brk_first:\n"
    "	brk #0\n"
    "	ret\n"
    ".size __trap_a64_brk_first, .-__trap_a64_brk_first\n");

TEST_CASE("Trap backend (aarch64): emulated first instructions")
{
	trap_attach_impl man;
	int hits = 0;
	auto cb = [&](const pt_regs &) { hits++; };
	REQUIRE(man.create_uprobe_at((void *)&__trap_a64_adrp_first, cb) >= 0);
	REQUIRE(__trap_a64_adrp_first() == &g_trap_global);
	REQUIRE(hits == 1);

	REQUIRE(man.create_uprobe_at((void *)&__trap_a64_b_first, cb) >= 0);
	REQUIRE(__trap_a64_b_first(8) == 50);
	REQUIRE(hits == 2);

	REQUIRE(man.create_uprobe_at((void *)&__trap_a64_cbz_first, cb) >= 0);
	REQUIRE(__trap_a64_cbz_first(0) == 2);
	REQUIRE(__trap_a64_cbz_first(9) == 1);
	REQUIRE(hits == 4);

	REQUIRE(man.create_uprobe_at((void *)&__trap_a64_tbnz_first, cb) >= 0);
	REQUIRE(__trap_a64_tbnz_first(2) == 1);
	REQUIRE(__trap_a64_tbnz_first(3) == 2);
	REQUIRE(hits == 6);

	REQUIRE(man.create_uprobe_at((void *)&__trap_a64_ldr_lit_first, cb) >=
		0);
	REQUIRE(__trap_a64_ldr_lit_first() == 0x1122334455667788ull);
	REQUIRE(hits == 7);

	REQUIRE(man.create_uprobe_at((void *)&__trap_a64_tailcall_first, cb) >=
		0);
	REQUIRE(man.create_uretprobe_at((void *)&__trap_a64_tailcall_first,
					[&](const pt_regs &regs) {
						REQUIRE(PT_REGS_RC(&regs) ==
							1005);
						hits++;
					}) >= 0);
	REQUIRE(__trap_a64_tailcall_first(5) == 1005);
	REQUIRE(hits == 9);

	REQUIRE(man.create_uprobe_at((void *)&__trap_a64_br_first, cb) >= 0);
	REQUIRE(__trap_a64_br_first(6, __trap_call_helper) == 1006);
	REQUIRE(hits == 10);
}

TEST_CASE("Trap backend (aarch64): refuses existing breakpoints")
{
	REQUIRE(check_probe_target((void *)&__trap_a64_brk_first).has_value());
	REQUIRE(check_probe_target(nullptr).has_value());
	REQUIRE(check_probe_target((const void *)&__trap_call_helper) ==
		std::nullopt);
}
#else
TEST_CASE("Trap backend: probe target check")
{
	REQUIRE(check_probe_target(nullptr).has_value());
	REQUIRE(check_probe_target((const void *)&check_probe_target) ==
		std::nullopt);
}
#endif
