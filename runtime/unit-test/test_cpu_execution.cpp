#include "bpftime_helper_group.hpp"
#include "bpftime_prog.hpp"
#include "bpftime_shm.hpp"
#include "bpftime_shm_internal.hpp"
#include "platform_utils.hpp"
#include <atomic>
#include <barrier>
#include <boost/interprocess/shared_memory_object.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <iterator>
#include <linux/filter.h>
#include <sched.h>
#include <string>
#include <thread>
#include <unistd.h>

using namespace bpftime;

namespace
{
std::atomic<int> active_helpers;
std::atomic<int> maximum_active_helpers;

uint64_t overlap_helper(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)
{
	int active = active_helpers.fetch_add(1, std::memory_order_acq_rel) + 1;
	int maximum = maximum_active_helpers.load(std::memory_order_relaxed);
	while (active > maximum &&
	       !maximum_active_helpers.compare_exchange_weak(
		       maximum, active, std::memory_order_relaxed)) {
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(1));
	active_helpers.fetch_sub(1, std::memory_order_release);
	return 0;
}
} // namespace

TEST_CASE("BPF executions on one CPU are serialized", "[percpu][execution]")
{
	const std::string shm_name = "bpftime_cpu_exec_test_" +
				     std::to_string(getpid());
	REQUIRE(setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1) == 0);
	boost::interprocess::shared_memory_object::remove(shm_name.c_str());
	bpftime_initialize_global_shm(shm_open_type::SHM_CREATE_ONLY);
	struct cleanup {
		const std::string &name;
		~cleanup()
		{
			bpftime_destroy_global_shm();
			boost::interprocess::shared_memory_object::remove(name.c_str());
			unsetenv("BPFTIME_GLOBAL_SHM_NAME");
		}
	} cleanup{ shm_name };

	struct bpf_insn insns[] = {
		BPF_EMIT_CALL(500),
		BPF_EXIT_INSN(),
	};
	bpftime_prog prog(reinterpret_cast<const ebpf_inst *>(insns),
			  std::size(insns), "per_cpu_serialization");
	bpftime_helper_info helper{ 500, "overlap_helper",
				    reinterpret_cast<void *>(overlap_helper) };
	REQUIRE(prog.bpftime_prog_register_raw_helper(std::move(helper)) == 0);
	REQUIRE(prog.bpftime_prog_load(true) == 0);

	cpu_set_t allowed;
	CPU_ZERO(&allowed);
	REQUIRE(sched_getaffinity(0, sizeof(allowed), &allowed) == 0);
	int cpu = 0;
	while (cpu < CPU_SETSIZE && !CPU_ISSET(cpu, &allowed))
		++cpu;
	REQUIRE(cpu < CPU_SETSIZE);

	active_helpers = 0;
	maximum_active_helpers = 0;
	std::atomic<int> failures = 0;
	std::barrier start(3);
	auto run = [&] {
		cpu_set_t affinity;
		CPU_ZERO(&affinity);
		CPU_SET(cpu, &affinity);
		if (sched_setaffinity(0, sizeof(affinity), &affinity) != 0)
			failures.fetch_add(1, std::memory_order_relaxed);
		start.arrive_and_wait();
		for (int i = 0; i < 16; ++i) {
			uint64_t result = 0;
			uint64_t context = 0;
			if (prog.bpftime_prog_exec(&context, sizeof(context), &result) !=
			    0)
				failures.fetch_add(1, std::memory_order_relaxed);
		}
	};
	std::thread first(run);
	std::thread second(run);
	start.arrive_and_wait();
	first.join();
	second.join();

	REQUIRE(failures == 0);
	REQUIRE(maximum_active_helpers == 1);

	// A timed-out contender must skip the probe instead of running outside
	// the lock and racing the program that still owns this CPU slot.
	maximum_active_helpers = 0;
	auto &shm = bpftime::shm_holder.global_shared_memory;
	shm.lock_bpf_cpu(cpu);
	int blocked_result = 0;
	std::thread blocked([&] {
		cpu_set_t affinity;
		CPU_ZERO(&affinity);
		CPU_SET(cpu, &affinity);
		if (sched_setaffinity(0, sizeof(affinity), &affinity) != 0) {
			blocked_result = -errno;
			return;
		}
		uint64_t result = 0, context = 0;
		blocked_result = prog.bpftime_prog_exec(&context, sizeof(context), &result);
	});
	blocked.join();
	shm.unlock_bpf_cpu(cpu);
	REQUIRE(blocked_result == -EAGAIN);
	REQUIRE(maximum_active_helpers == 0);
}

TEST_CASE("Per-CPU try-lock degrades instead of hanging or throwing",
	  "[percpu][locking]")
{
	const std::string shm_name = "bpftime_cpu_lock_test_" +
				     std::to_string(getpid());
	REQUIRE(setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1) == 0);
	boost::interprocess::shared_memory_object::remove(shm_name.c_str());
	bpftime_initialize_global_shm(shm_open_type::SHM_CREATE_ONLY);
	struct cleanup {
		const std::string &name;
		~cleanup()
		{
			bpftime_destroy_global_shm();
			boost::interprocess::shared_memory_object::remove(
				name.c_str());
			unsetenv("BPFTIME_GLOBAL_SHM_NAME");
		}
	} cleanup{ shm_name };

	auto &shm = bpftime::shm_holder.global_shared_memory;
	constexpr int cpu = 0;

	// Uncontended acquisition succeeds and releases cleanly.
	REQUIRE(shm.try_lock_bpf_cpu(cpu));
	shm.unlock_bpf_cpu(cpu);

	// Hold the slot on this thread; a concurrent bounded attempt must give
	// up (return false) rather than block forever or throw into the host.
	shm.lock_bpf_cpu(cpu);
	std::atomic<bool> acquired{ true };
	std::thread contender([&] {
		acquired = shm.try_lock_bpf_cpu(cpu);
	});
	contender.join();
	REQUIRE_FALSE(acquired.load());
	shm.unlock_bpf_cpu(cpu);

	// After release the bounded lock is obtainable again.
	REQUIRE(shm.try_lock_bpf_cpu(cpu));
	shm.unlock_bpf_cpu(cpu);
}
