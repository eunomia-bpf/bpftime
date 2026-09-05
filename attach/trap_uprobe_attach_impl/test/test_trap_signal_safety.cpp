#include "trap_test_common.hpp"
#include <trap_uprobe_attach_impl.hpp>
#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace bpftime;
using namespace bpftime::attach::trap;

// Wraps malloc so the probe sits on our symbol, not libc's.
extern "C" TRAP_TEST_TARGET void *__trap_ss_alloc(size_t n)
{
	asm("");
	void *p = malloc(n);
	asm("");
	return p;
}

extern "C" TRAP_TEST_TARGET void __trap_ss_free_wrap(void *p)
{
	asm("");
	free(p);
	asm("");
}

TEST_CASE("Trap backend: handler signal-safety under allocator stress")
{
	trap_attach_impl man;
	std::atomic<int> uprobe_hits{ 0 };
	std::atomic<int> uretprobe_hits{ 0 };
	std::atomic<bool> stop{ false };

	REQUIRE(man.create_uprobe_at(
			(void *)&__trap_ss_alloc,
			[&](const pt_regs &) {
				uprobe_hits.fetch_add(1,
						     std::memory_order_relaxed);
			}) >= 0);
	REQUIRE(man.create_uretprobe_at(
			(void *)&__trap_ss_alloc,
			[&](const pt_regs &) {
				uretprobe_hits.fetch_add(
					1, std::memory_order_relaxed);
			}) >= 0);

	constexpr int N_THREADS = 4;
	std::vector<std::thread> threads;
	for (int i = 0; i < N_THREADS; i++) {
		threads.emplace_back([&] {
			while (!stop.load(std::memory_order_relaxed)) {
				void *p = __trap_ss_alloc(64);
				__trap_ss_free_wrap(p);
			}
		});
	}

	// Run long enough that every thread's first uretprobe hit triggers
	// lazy uret_stack allocation inside the handler.
	std::this_thread::sleep_for(std::chrono::milliseconds(500));
	stop.store(true, std::memory_order_relaxed);

	for (auto &t : threads)
		t.join();

	REQUIRE(uprobe_hits.load() > 0);
	REQUIRE(uretprobe_hits.load() > 0);
	REQUIRE(uprobe_hits.load() == uretprobe_hits.load());
}

TEST_CASE("Trap backend: new threads get uret stacks without deadlock")
{
	trap_attach_impl man;
	std::atomic<int> hits{ 0 };
	std::atomic<bool> stop{ false };

	REQUIRE(man.create_uprobe_at(
			(void *)&__trap_ss_alloc,
			[&](const pt_regs &) {
				hits.fetch_add(1, std::memory_order_relaxed);
			}) >= 0);
	REQUIRE(man.create_uretprobe_at(
			(void *)&__trap_ss_alloc,
			[&](const pt_regs &) {
				hits.fetch_add(1, std::memory_order_relaxed);
			}) >= 0);

	// Spawn waves of short-lived threads, each of which must lazily
	// allocate its uret stack on the very first probe hit.
	for (int wave = 0; wave < 4; wave++) {
		std::vector<std::thread> batch;
		for (int i = 0; i < 8; i++) {
			batch.emplace_back([&] {
				for (int j = 0; j < 50; j++) {
					void *p = __trap_ss_alloc(32);
					__trap_ss_free_wrap(p);
				}
			});
		}
		for (auto &t : batch)
			t.join();
	}

	REQUIRE(hits.load() > 0);
}
