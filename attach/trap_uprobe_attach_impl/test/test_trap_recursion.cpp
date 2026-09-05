#include "trap_test_common.hpp"
#include <trap_uprobe_attach_impl.hpp>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <thread>
#include <vector>

using namespace bpftime;
using namespace bpftime::attach::trap;

extern "C" TRAP_TEST_TARGET uint64_t __trap_reent_target(uint64_t a)
{
	asm("");
	return a + 5;
}

// A callback that calls the probed function must not recurse into the probe
// forever; nested hits run the original code without callbacks.
TEST_CASE("Trap backend: callbacks may call the probed function")
{
	trap_attach_impl man;
	int hits = 0;
	uint64_t nested_result = 0;
	REQUIRE(man.create_uprobe_at((void *)&__trap_reent_target,
				     [&](const pt_regs &regs) {
					     hits++;
					     nested_result = __trap_reent_target(
						     PT_REGS_PARM1(&regs) * 10);
				     }) >= 0);
	REQUIRE(__trap_reent_target(1) == 6);
	REQUIRE(hits == 1);
	REQUIRE(nested_result == 15);
}

// Probing malloc exercises the async-signal-safety of the handler: the
// callback allocates, new threads hit the probe for the first time, and
// nested allocations inside the handler must not deadlock.
TEST_CASE("Trap backend: probing malloc from several threads")
{
	void *malloc_addr = dlsym(RTLD_DEFAULT, "malloc");
	REQUIRE(malloc_addr != nullptr);
	trap_attach_impl man;
	std::atomic<uint64_t> hits{ 0 };
	std::atomic<uint64_t> rets{ 0 };
	int id = man.create_uprobe_at(malloc_addr, [&](const pt_regs &regs) {
		hits.fetch_add(PT_REGS_PARM1(&regs) == 12345 ? 1 : 0,
			       std::memory_order_relaxed);
		// Allocating inside the callback re-enters malloc
		void *p = malloc(64);
		free(p);
	});
	if (id < 0) {
		// Some libc builds start malloc with an instruction this
		// backend refuses to relocate; that is a supported outcome
		WARN("malloc could not be probed on this platform");
		return;
	}
	int rid = man.create_uretprobe_at(malloc_addr, [&](const pt_regs &regs) {
		if (PT_REGS_RC(&regs) != 0)
			rets.fetch_add(1, std::memory_order_relaxed);
	});
	REQUIRE(rid >= 0);
	std::vector<std::thread> threads;
	for (int t = 0; t < 4; t++) {
		threads.emplace_back([&]() {
			for (int i = 0; i < 1000; i++) {
				void *p = malloc(12345);
				REQUIRE(p != nullptr);
				free(p);
			}
		});
	}
	for (auto &th : threads)
		th.join();
	REQUIRE(hits.load() >= 4000);
	REQUIRE(rets.load() >= 4000);
	REQUIRE(man.detach_by_id(id) == 0);
	REQUIRE(man.detach_by_id(rid) == 0);
	void *p = malloc(12345);
	REQUIRE(p != nullptr);
	free(p);
}
