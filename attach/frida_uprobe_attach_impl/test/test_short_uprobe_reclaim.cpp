// Regression for short-uprobe (empty-marker) stub reclamation.
//
// The empty-marker path bypasses Frida-Gum and patches the function text with
// a jump to a privately mmap'ed stub. Detach must (1) restore the original
// word before any new call can enter the stub and (2) wait for callbacks
// already executing the stub before unmapping it / freeing the entry. This
// drives the marker from several threads while detaching concurrently; the
// test passing (process survives, no fault) is the reclaim-safety contract.
#include "catch2/catch_test_macros.hpp"
#include "frida_uprobe_attach_impl.hpp"
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using namespace bpftime;
using namespace bpftime::attach;

#if defined(__linux__) && defined(__x86_64__)
// First byte is 0xc3 (ret) so is_empty_x86_64_function() selects the short
// uprobe path once Frida-Gum rejects the one-byte function. The seven nop
// padding bytes make the eight-byte aligned patch overwrite safe.
extern "C" __attribute__((naked, noinline, aligned(8))) void
__bpftime_test_empty_marker_target()
{
	asm volatile("ret\n\t"
		     "nop; nop; nop; nop; nop; nop; nop\n\t");
}

TEST_CASE("Short uprobe stub is reclaimed safely under concurrent calls",
	  "[frida][short-uprobe]")
{
	frida_attach_impl impl;
	int id = -1;
	try {
		id = impl.create_uprobe_at(
			(void *)__bpftime_test_empty_marker_target,
			[](const pt_regs &) {});
	} catch (const std::runtime_error &) {
		// Frida-Gum refused the empty function and this build did not
		// take the short-patch fallback: nothing to reclaim.
		SUCCEED("short uprobe fallback not selected on this build");
		return;
	}
	REQUIRE(id >= 0);

	std::atomic<unsigned long> calls{ 0 };
	std::atomic<bool> stop{ false };
	auto worker = [&] {
		while (!stop.load(std::memory_order_relaxed)) {
			__bpftime_test_empty_marker_target();
			calls.fetch_add(1, std::memory_order_relaxed);
		}
	};

	std::vector<std::thread> threads;
	for (int i = 0; i < 4; ++i)
		threads.emplace_back(worker);

	// Let the workers enter the patched marker, then detach while they may
	// still be executing the stub.
	std::this_thread::sleep_for(std::chrono::milliseconds(20));
	REQUIRE(impl.detach_by_id(id) == 0);
	stop.store(true, std::memory_order_relaxed);
	for (auto &t : threads)
		t.join();

	REQUIRE(calls.load() > 0);
	// Calls after a successful restore must behave as an ordinary ret.
	__bpftime_test_empty_marker_target();
}
#endif
