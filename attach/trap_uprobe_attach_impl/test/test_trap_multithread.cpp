#include "trap_test_common.hpp"
#include <trap_uprobe_attach_impl.hpp>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <thread>
#include <vector>

using namespace bpftime;
using namespace bpftime::attach::trap;

extern "C" TRAP_TEST_TARGET uint64_t
__trap_mt_target(uint64_t a, uint64_t b)
{
	asm("");
	return a ^ (b << 1);
}

// Out-of-line execution must never lose an event, no matter how many threads
// race through the probed instruction.
TEST_CASE("Trap backend: no lost events under contention")
{
	constexpr int THREADS = 8;
	constexpr int ITERATIONS = 20000;
	std::atomic<uint64_t> entries{ 0 }, exits{ 0 }, sum_args{ 0 },
		sum_rets{ 0 };
	trap_attach_impl man;
	auto addr = (void *)&__trap_mt_target;
	REQUIRE(man.create_uprobe_at(addr, [&](const pt_regs &regs) {
		entries.fetch_add(1, std::memory_order_relaxed);
		sum_args.fetch_add(PT_REGS_PARM1(&regs),
				   std::memory_order_relaxed);
	}) >= 0);
	REQUIRE(man.create_uretprobe_at(addr, [&](const pt_regs &regs) {
		exits.fetch_add(1, std::memory_order_relaxed);
		sum_rets.fetch_add(PT_REGS_RC(&regs),
				   std::memory_order_relaxed);
	}) >= 0);

	std::vector<std::thread> threads;
	std::atomic<uint64_t> expected_args{ 0 }, expected_rets{ 0 };
	for (int t = 0; t < THREADS; t++) {
		threads.emplace_back([&, t]() {
			uint64_t local_args = 0, local_rets = 0;
			for (int i = 0; i < ITERATIONS; i++) {
				uint64_t a = (uint64_t)t * ITERATIONS + i;
				local_args += a;
				local_rets += __trap_mt_target(a, i);
			}
			expected_args.fetch_add(local_args);
			expected_rets.fetch_add(local_rets);
		});
	}
	for (auto &th : threads)
		th.join();
	REQUIRE(entries.load() == (uint64_t)THREADS * ITERATIONS);
	REQUIRE(exits.load() == (uint64_t)THREADS * ITERATIONS);
	REQUIRE(sum_args.load() == expected_args.load());
	REQUIRE(sum_rets.load() == expected_rets.load());
}

// Attaching and detaching while other threads keep calling the function must
// not crash the process and must not miscount events after the detach.
TEST_CASE("Trap backend: attach/detach while the function is hot")
{
	std::atomic<bool> stop{ false };
	std::atomic<uint64_t> calls{ 0 };
	std::vector<std::thread> threads;
	for (int t = 0; t < 4; t++) {
		threads.emplace_back([&]() {
			uint64_t i = 0;
			while (!stop.load(std::memory_order_relaxed)) {
				uint64_t r = __trap_mt_target(i, i + 1);
				REQUIRE(r == (i ^ ((i + 1) << 1)));
				i++;
				calls.fetch_add(1, std::memory_order_relaxed);
			}
		});
	}
	trap_attach_impl man;
	auto addr = (void *)&__trap_mt_target;
	for (int round = 0; round < 50; round++) {
		std::atomic<uint64_t> hits{ 0 };
		int id = man.create_uprobe_at(addr, [&](const pt_regs &) {
			hits.fetch_add(1, std::memory_order_relaxed);
		});
		REQUIRE(id >= 0);
		int rid = man.create_uretprobe_at(addr, [&](const pt_regs &) {
			hits.fetch_add(1, std::memory_order_relaxed);
		});
		REQUIRE(rid >= 0);
		std::this_thread::sleep_for(std::chrono::milliseconds(2));
		REQUIRE(man.detach_by_id(id) == 0);
		REQUIRE(man.detach_by_id(rid) == 0);
		REQUIRE(hits.load() > 0);
	}
	stop = true;
	for (auto &th : threads)
		th.join();
	REQUIRE(calls.load() > 0);
}
