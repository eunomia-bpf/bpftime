#include "trap_test_common.hpp"
#include <trap_uprobe_attach_impl.hpp>
#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <thread>
#include <vector>

using namespace bpftime;
using namespace bpftime::attach::trap;

extern "C" TRAP_TEST_TARGET uint64_t __trap_cp_target(uint64_t a)
{
	asm("");
	return a + 1;
}

#if defined(__riscv) && __riscv_xlen == 64
// A function whose second instruction is 4 bytes long and sits at
// addr ≡ 2 (mod 4): the leading c.nop is emitted compressed and takes 2
// bytes, and `norvc` keeps the addi that follows from being compressed to
// c.addi. Probing that addi — not the function entry — is what exercises the
// three-phase write path under concurrent execution.
extern "C" uint64_t __trap_cp_misaligned4(uint64_t);
asm(".option push\n"
    ".option rvc\n"
    ".balign 4\n"
    ".globl __trap_cp_misaligned4\n"
    ".type __trap_cp_misaligned4, @function\n"
    "__trap_cp_misaligned4:\n"
    "	c.nop\n"
    ".option push\n"
    ".option norvc\n"
    "	addi a0, a0, 1\n"
    ".option pop\n"
    "	ret\n"
    ".size __trap_cp_misaligned4, .-__trap_cp_misaligned4\n"
    ".option pop\n");
#endif

TEST_CASE("Trap backend: concurrent arm/disarm does not crash")
{
	constexpr int N_THREADS = 4;
	constexpr int N_CYCLES = 200;
	std::atomic<bool> stop{ false };
	std::atomic<uint64_t> call_count{ 0 };

	// Worker threads continuously call the target function.
	std::vector<std::thread> workers;
	for (int i = 0; i < N_THREADS; i++) {
		workers.emplace_back([&] {
			while (!stop.load(std::memory_order_relaxed)) {
				volatile uint64_t r = __trap_cp_target(42);
				(void)r;
				call_count.fetch_add(
					1, std::memory_order_relaxed);
			}
		});
	}

	// Main thread repeatedly creates and destroys probes while
	// workers are executing the target. Destruction triggers
	// write_code to restore the original instruction — the
	// three-phase protocol must keep every intermediate state safe.
	for (int i = 0; i < N_CYCLES; i++) {
		trap_attach_impl man;
		std::atomic<int> hits{ 0 };
		REQUIRE(man.create_uprobe_at(
				(void *)&__trap_cp_target,
				[&](const pt_regs &) {
					hits.fetch_add(
						1, std::memory_order_relaxed);
				}) >= 0);
		// Let some hits accumulate before disarming
		for (int spin = 0; spin < 200; spin++)
			sched_yield();
	}

	stop.store(true, std::memory_order_relaxed);
	for (auto &t : workers)
		t.join();

	REQUIRE(call_count.load() > 0);
	// After all arm/disarm cycles the function must still work
	REQUIRE(__trap_cp_target(99) == 100);
}

TEST_CASE("Trap backend: concurrent arm/disarm with uretprobe")
{
	constexpr int N_THREADS = 4;
	constexpr int N_CYCLES = 100;
	std::atomic<bool> stop{ false };
	std::atomic<uint64_t> call_count{ 0 };

	std::vector<std::thread> workers;
	for (int i = 0; i < N_THREADS; i++) {
		workers.emplace_back([&] {
			while (!stop.load(std::memory_order_relaxed)) {
				volatile uint64_t r = __trap_cp_target(10);
				(void)r;
				call_count.fetch_add(
					1, std::memory_order_relaxed);
			}
		});
	}

	for (int i = 0; i < N_CYCLES; i++) {
		trap_attach_impl man;
		std::atomic<int> entry_hits{ 0 };
		std::atomic<int> exit_hits{ 0 };
		REQUIRE(man.create_uprobe_at(
				(void *)&__trap_cp_target,
				[&](const pt_regs &) {
					entry_hits.fetch_add(
						1, std::memory_order_relaxed);
				}) >= 0);
		REQUIRE(man.create_uretprobe_at(
				(void *)&__trap_cp_target,
				[&](const pt_regs &) {
					exit_hits.fetch_add(
						1, std::memory_order_relaxed);
				}) >= 0);
		for (int spin = 0; spin < 200; spin++)
			sched_yield();
	}

	stop.store(true, std::memory_order_relaxed);
	for (auto &t : workers)
		t.join();

	REQUIRE(call_count.load() > 0);
	REQUIRE(__trap_cp_target(99) == 100);
}

#if defined(__riscv) && __riscv_xlen == 64
TEST_CASE("Trap backend: concurrent three-phase patch (4-byte insn at "
	   "addr % 4 == 2)")
{
	// The probe goes on the addi, not on the function entry: the entry
	// holds the 2-byte c.nop, which would be patched with a single aligned
	// 16-bit store and would never reach the three-phase path.
	auto *fn = (uint8_t *)__trap_cp_misaligned4;
	auto *probe_addr = fn + 2;
	REQUIRE(((uintptr_t)probe_addr & 3) == 2);
	// On RISC-V an instruction that is not compressed has its two low bits
	// set; this fails loudly if the assembler ever compresses the addi.
	REQUIRE((probe_addr[0] & 0x3) == 0x3);
	REQUIRE(check_probe_target(probe_addr) == std::nullopt);

	const uint64_t splits_before = split_write_count();

	constexpr int N_THREADS = 4;
	constexpr int N_CYCLES = 200;
	std::atomic<bool> stop{ false };
	std::atomic<uint64_t> call_count{ 0 };
	std::atomic<int> ready{ 0 };
	std::atomic<uint64_t> worker_hits[N_THREADS]{};
	// Catch2 assertions are not thread safe, so the workers only record
	// and the result is checked after the join.
	std::atomic<bool> bad_result{ false };

	std::vector<std::thread> workers;
	for (int i = 0; i < N_THREADS; i++) {
		workers.emplace_back([&, i] {
			ready.fetch_add(1);
			while (!stop.load(std::memory_order_relaxed)) {
				if (__trap_cp_misaligned4(42 + i) != (uint64_t)(43 + i))
					bad_result.store(
						true,
						std::memory_order_relaxed);
				call_count.fetch_add(
					1, std::memory_order_relaxed);
			}
		});
	}

	while (ready.load() != N_THREADS)
		sched_yield();

	std::atomic<int> hits{ 0 };
	for (int i = 0; i < N_CYCLES; i++) {
		trap_attach_impl man;
		REQUIRE(man.create_uprobe_at(
				(void *)probe_addr,
				[&](const pt_regs &) {
					uint64_t argument = 0;
					if (bpftime_trap_get_func_arg(0, 0, &argument, 0, 0) == 0 &&
					    argument >= 42 && argument < 42 + N_THREADS)
						worker_hits[argument - 42].fetch_add(1, std::memory_order_relaxed);
					else
						bad_result.store(true, std::memory_order_relaxed);
					hits.fetch_add(
						1, std::memory_order_relaxed);
				}) >= 0);
		for (int spin = 0; spin < 200; spin++)
			sched_yield();
	}

	stop.store(true, std::memory_order_relaxed);
	for (auto &t : workers)
		t.join();

	REQUIRE(call_count.load() > 0);
	REQUIRE_FALSE(bad_result.load());
	// Every cycle arms and disarms once, so the three-phase protocol must
	// have run. Without this the test can silently patch a compressed
	// instruction instead and assert nothing about the path under test.
	REQUIRE(split_write_count() == splits_before + 2 * N_CYCLES);
	INFO("split writes=" << split_write_count() - splits_before
	     << " hits=" << hits.load() << " calls=" << call_count.load());
	REQUIRE(hits.load() > 0);
	for (const auto &count : worker_hits)
		REQUIRE(count.load() > 0);
	REQUIRE(__trap_cp_misaligned4(99) == 100);
}
#endif
