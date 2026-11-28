#include "catch2/catch_test_macros.hpp"
#include "frida_uprobe_attach_impl.hpp"
#include <atomic>
#include <base_attach_impl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

using namespace bpftime::attach;
using namespace bpftime;

namespace
{
struct reset_process_suppression_guard {
	~reset_process_suppression_guard()
	{
		process_attach_callbacks_suppressed.store(
			false, std::memory_order_seq_cst);
	}
};

extern "C" __attribute__((noinline)) int
bpftime_test_exit_quiesce_target(int x)
{
	asm volatile("" : "+r"(x) : : "memory");
	return x + 7;
}
} // namespace

TEST_CASE("Test bpftime_set_retval")
{
	curr_thread_override_return_callback.reset();
	REQUIRE(bpftime_set_retval(0) == static_cast<uint64_t>(-EINVAL));
	bool called = false;
	curr_thread_override_return_callback = [&](uint64_t, uint64_t) {
		called = true;
	};
	REQUIRE(bpftime_set_retval(0) == 0);
	REQUIRE(called);
}

TEST_CASE("Test bpftime_override_return")
{
	curr_thread_override_return_callback.reset();
	REQUIRE(bpftime_override_return(0, 0) ==
		static_cast<uint64_t>(-EINVAL));

	bool called = false;
	curr_thread_override_return_callback = [&](uint64_t, uint64_t) {
		called = true;
	};
	REQUIRE(bpftime_override_return(0, 0) == 0);
	REQUIRE(called);
}

TEST_CASE("Process-wide callback suppression quiesces active Frida hooks")
{
	reset_process_suppression_guard reset_guard;
	process_attach_callbacks_suppressed.store(false,
						  std::memory_order_release);

	{
		attach_callback_scope scope;
		REQUIRE(scope.entered());
	}

	frida_attach_impl impl;
	std::atomic<uint64_t> uprobe_hits{ 0 };
	std::atomic<uint64_t> uretprobe_hits{ 0 };
	int id_enter = impl.create_uprobe_at(
		(void *)&bpftime_test_exit_quiesce_target,
		[&](const pt_regs &) {
			uprobe_hits.fetch_add(1, std::memory_order_relaxed);
		});
	int id_leave = impl.create_uretprobe_at(
		(void *)&bpftime_test_exit_quiesce_target,
		[&](const pt_regs &) {
			uretprobe_hits.fetch_add(1, std::memory_order_relaxed);
		});
	REQUIRE(id_enter >= 0);
	REQUIRE(id_leave >= 0);
	REQUIRE(bpftime_test_exit_quiesce_target(5) == 12);
	REQUIRE(uprobe_hits.load(std::memory_order_relaxed) == 1);
	REQUIRE(uretprobe_hits.load(std::memory_order_relaxed) == 1);

	process_attach_callbacks_suppressed.store(true,
						  std::memory_order_release);
	REQUIRE(bpftime_test_exit_quiesce_target(10) == 17);
	REQUIRE(uprobe_hits.load(std::memory_order_relaxed) == 1);
	REQUIRE(uretprobe_hits.load(std::memory_order_relaxed) == 1);

	{
		attach_callback_scope suppressed_scope;
		REQUIRE_FALSE(suppressed_scope.entered());
	}

	REQUIRE(impl.detach_by_id(id_enter) == 0);
	REQUIRE(impl.detach_by_id(id_leave) == 0);
}

TEST_CASE("Detached background threads calling hooked target survive std::exit")
{
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		alarm(15);
		auto *impl = new frida_attach_impl();
		std::atomic<uint64_t> hits{ 0 };
		impl->create_uprobe_at(
			(void *)&bpftime_test_exit_quiesce_target,
			[&](const pt_regs &) {
				hits.fetch_add(1, std::memory_order_relaxed);
			});
		impl->create_uretprobe_at(
			(void *)&bpftime_test_exit_quiesce_target,
			[&](const pt_regs &) {
				hits.fetch_add(1, std::memory_order_relaxed);
			});
		REQUIRE(std::atexit([]() {
				process_attach_callbacks_suppressed.store(
					true, std::memory_order_release);
			}) == 0);
		for (int i = 0; i < 4; ++i) {
			std::thread([i]() {
				for (uint64_t iter = 0;; ++iter) {
					(void)bpftime_test_exit_quiesce_target(
						static_cast<int>(iter) + i);
				}
			}).detach();
		}
		while (hits.load(std::memory_order_relaxed) < 500) {
			std::this_thread::yield();
		}
		std::exit(0);
	}
	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE(WIFEXITED(status));
	REQUIRE(WEXITSTATUS(status) == 0);
}
