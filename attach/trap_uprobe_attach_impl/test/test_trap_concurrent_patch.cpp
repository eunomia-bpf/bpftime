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
