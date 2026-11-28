#include "bpf_map/userspace/per_cpu_array_map.hpp"
#include "bpftime_helper_group.hpp"
#include "bpftime_prog.hpp"
#include "bpftime_shm.hpp"
#include "bpftime_shm_internal.hpp"
#include "platform_utils.hpp"
#include <algorithm>
#include <atomic>
#include <barrier>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <ebpf_inst.h>
#include <iterator>
#include <sched.h>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

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
	std::this_thread::sleep_for(std::chrono::milliseconds(5));
	active_helpers.fetch_sub(1, std::memory_order_release);
	return 0;
}
} // namespace

TEST_CASE("BPF executions on one CPU do not block or serialize each other",
	  "[percpu][execution]")
{
	const std::string shm_name =
		"bpftime_cpu_exec_test_" + std::to_string(getpid());
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

	const ebpf_inst insns[] = {
		{ EBPF_OP_CALL, 0, 0, 0, 500 },
		{ EBPF_OP_EXIT, 0, 0, 0, 0 },
	};
	bpftime_prog prog(insns, std::size(insns), "non_blocking_cpu_exec");
	bpftime_helper_info helper{ 500, "overlap_helper",
				    reinterpret_cast<void *>(overlap_helper) };
	REQUIRE(prog.bpftime_prog_register_raw_helper(std::move(helper)) == 0);
	REQUIRE(prog.bpftime_prog_load(true) == 0);

#ifdef __linux__
	cpu_set_t allowed;
	CPU_ZERO(&allowed);
	REQUIRE(sched_getaffinity(0, sizeof(allowed), &allowed) == 0);
	int cpu = 0;
	while (cpu < CPU_SETSIZE && !CPU_ISSET(cpu, &allowed))
		++cpu;
	REQUIRE(cpu < CPU_SETSIZE);
#endif

	active_helpers = 0;
	maximum_active_helpers = 0;
	std::atomic<int> failures = 0;
	std::barrier start(3);
	auto run = [&] {
#ifdef __linux__
		cpu_set_t affinity;
		CPU_ZERO(&affinity);
		CPU_SET(cpu, &affinity);
		if (sched_setaffinity(0, sizeof(affinity), &affinity) != 0)
			failures.fetch_add(1, std::memory_order_relaxed);
#endif
		start.arrive_and_wait();
		for (int i = 0; i < 4; ++i) {
			uint64_t result = 0;
			uint64_t context = 0;
			if (prog.bpftime_prog_exec(&context, sizeof(context),
						   &result) != 0)
				failures.fetch_add(1,
						   std::memory_order_relaxed);
		}
	};
	std::thread first(run);
	std::thread second(run);
	start.arrive_and_wait();
	first.join();
	second.join();

	REQUIRE(failures == 0);
	REQUIRE(maximum_active_helpers >= 2);
}

#ifdef __linux__
namespace
{
int migrate_target_cpu = -1;

uint64_t cpu_pin_probe_helper(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)
{
	int first = my_sched_getcpu();
	if (migrate_target_cpu >= 0) {
		cpu_set_t target;
		CPU_ZERO(&target);
		CPU_SET(migrate_target_cpu, &target);
		(void)sched_setaffinity(0, sizeof(target), &target);
	}
	int second = my_sched_getcpu();
	return (static_cast<uint64_t>(static_cast<uint32_t>(first)) << 32) |
	       static_cast<uint32_t>(second);
}
} // namespace

TEST_CASE("Lazy CPU ID stays pinned within one BPF call and refreshes on next",
	  "[percpu][execution]")
{
	cpu_set_t allowed;
	CPU_ZERO(&allowed);
	REQUIRE(sched_getaffinity(0, sizeof(allowed), &allowed) == 0);
	int cpu_a = -1, cpu_b = -1;
	for (int c = 0; c < CPU_SETSIZE; ++c) {
		if (!CPU_ISSET(c, &allowed))
			continue;
		if (cpu_a < 0)
			cpu_a = c;
		else {
			cpu_b = c;
			break;
		}
	}
	if (cpu_b < 0)
		return; // Single-CPU environment cannot test migration.

	struct restore_affinity {
		cpu_set_t orig;
		~restore_affinity()
		{
			(void)sched_setaffinity(0, sizeof(orig), &orig);
		}
	} restore{ allowed };

	cpu_set_t start_set;
	CPU_ZERO(&start_set);
	CPU_SET(cpu_a, &start_set);
	REQUIRE(sched_setaffinity(0, sizeof(start_set), &start_set) == 0);

	bpftime_destroy_global_shm();
	bpftime_initialize_global_shm(shm_open_type::SHM_NO_CREATE);
	const ebpf_inst insns[] = {
		{ EBPF_OP_CALL, 0, 0, 0, 501 },
		{ EBPF_OP_EXIT, 0, 0, 0, 0 },
	};
	runtime_config config;
	config.set_vm_name(DEFAULT_VM_NAME);
	bpftime_prog prog(insns, std::size(insns), "lazy_cpu_pin",
			  std::move(config));
	bpftime_helper_info helper{
		501, "cpu_pin_probe_helper",
		reinterpret_cast<void *>(cpu_pin_probe_helper)
	};
	REQUIRE(prog.bpftime_prog_register_raw_helper(std::move(helper)) == 0);
	REQUIRE(prog.bpftime_prog_load(true) == 0);

	migrate_target_cpu = cpu_b;
	uint64_t ret = 0, ctx = 0;
	REQUIRE(prog.bpftime_prog_exec(&ctx, sizeof(ctx), &ret) == 0);
	REQUIRE(static_cast<int>(ret >> 32) == cpu_a);
	REQUIRE(static_cast<int>(ret & 0xffffffffU) == cpu_a);

	// Next invocation starts on cpu_b and must observe cpu_b.
	migrate_target_cpu = -1;
	REQUIRE(prog.bpftime_prog_exec(&ctx, sizeof(ctx), &ret) == 0);
	REQUIRE(static_cast<int>(ret >> 32) == cpu_b);
	REQUIRE(static_cast<int>(ret & 0xffffffffU) == cpu_b);
}
#endif

TEST_CASE("Map-free local programs run without shared memory",
	  "[percpu][execution]")
{
	bpftime_destroy_global_shm();
	bpftime_initialize_global_shm(shm_open_type::SHM_NO_CREATE);
	const ebpf_inst insns[] = {
		{ EBPF_OP_MOV64_IMM, 0, 0, 0, 42 },
		{ EBPF_OP_EXIT, 0, 0, 0, 0 },
	};
	runtime_config config;
	config.set_vm_name(DEFAULT_VM_NAME);
	bpftime_prog prog(insns, std::size(insns), "map_free_local",
			  std::move(config));
	REQUIRE(prog.bpftime_prog_load(true) == 0);
	uint64_t result = 0;
	REQUIRE(prog.bpftime_prog_exec(nullptr, 0, &result) == 0);
	REQUIRE(result == 42);
}

TEST_CASE("PERCPU_ARRAY helper paths reject out-of-range CPU index safely",
	  "[percpu][execution]")
{
	const std::string shm_name =
		"bpftime_percpu_bounds_" + std::to_string(getpid());
	boost::interprocess::shared_memory_object::remove(shm_name.c_str());
	struct cleanup_shm {
		const std::string &name;
		~cleanup_shm()
		{
			boost::interprocess::shared_memory_object::remove(
				name.c_str());
		}
	} cleanup_shm{ shm_name };
	boost::interprocess::managed_shared_memory mem(
		boost::interprocess::create_only, shm_name.c_str(), 1 << 20);
	per_cpu_array_map_impl map(mem, sizeof(uint64_t), 4, 0);
	uint32_t key = 0;
	uint64_t val = 99;
	errno = 0;
	REQUIRE(map.elem_lookup(&key) == nullptr);
	REQUIRE(errno == EINVAL);
	errno = 0;
	REQUIRE(map.elem_update(&key, &val, 0) == -1);
	REQUIRE(errno == EINVAL);
}

namespace
{
per_cpu_array_map_impl *g_scratch_map = nullptr;
std::atomic<uint64_t> g_scratch_corruptions{ 0 };
std::atomic<uint64_t> g_scratch_completed{ 0 };

uint64_t percpu_scratch_helper(uint64_t token, uint64_t, uint64_t, uint64_t,
			       uint64_t)
{
	uint32_t key = 0;
	auto *words =
		static_cast<uint64_t *>(g_scratch_map->elem_lookup(&key));
	if (!words)
		return 0;
	for (int w = 0; w < 64; ++w)
		words[w] = token ^ static_cast<uint64_t>(w);
	std::this_thread::sleep_for(std::chrono::microseconds(200));
	for (int w = 0; w < 64; ++w) {
		if (words[w] != (token ^ static_cast<uint64_t>(w))) {
			g_scratch_corruptions.fetch_add(
				1, std::memory_order_relaxed);
			break;
		}
	}
	g_scratch_completed.fetch_add(1, std::memory_order_relaxed);
	return 0;
}
} // namespace

TEST_CASE("Preempted threads on one physical CPU get distinct vCPU slots without scratch corruption",
	  "[percpu][execution]")
{
	const int ncpu = std::max<int>(1, sysconf(_SC_NPROCESSORS_ONLN));
	const int workers = std::min<int>(4, ncpu);
	const std::string shm_name =
		"bpftime_vcpu_scratch_" + std::to_string(getpid());
	const std::string map_shm_name = shm_name + "_map";
	REQUIRE(setenv("BPFTIME_GLOBAL_SHM_NAME", shm_name.c_str(), 1) == 0);
	// All workers share one CPU, so all but one run on extra slots.
	REQUIRE(setenv("BPFTIME_VCPU_RATIO", "4", 1) == 0);
	boost::interprocess::shared_memory_object::remove(shm_name.c_str());
	boost::interprocess::shared_memory_object::remove(map_shm_name.c_str());
	bpftime_initialize_global_shm(shm_open_type::SHM_CREATE_ONLY);
	struct cleanup {
		const std::string &name;
		const std::string &map_name;
		~cleanup()
		{
			g_scratch_map = nullptr;
			bpftime_destroy_global_shm();
			boost::interprocess::shared_memory_object::remove(
				map_name.c_str());
			boost::interprocess::shared_memory_object::remove(
				name.c_str());
			unsetenv("BPFTIME_GLOBAL_SHM_NAME");
			unsetenv("BPFTIME_VCPU_RATIO");
		}
	} cleanup{ shm_name, map_shm_name };

	boost::interprocess::managed_shared_memory map_mem(
		boost::interprocess::create_only, map_shm_name.c_str(),
		2 << 20);
	per_cpu_array_map_impl map(map_mem, 64 * sizeof(uint64_t), 1,
				   shm_holder.global_shared_memory.vcpu_count());
	g_scratch_map = &map;
	g_scratch_corruptions = 0;
	g_scratch_completed = 0;

	const ebpf_inst insns[] = {
		{ EBPF_OP_LDXW, 1, 1, 0, 0 },
		{ EBPF_OP_CALL, 0, 0, 0, 502 },
		{ EBPF_OP_EXIT, 0, 0, 0, 0 },
	};
	bpftime_prog prog(insns, std::size(insns), "vcpu_scratch_isolation");
	bpftime_helper_info helper{
		502, "percpu_scratch_helper",
		reinterpret_cast<void *>(percpu_scratch_helper)
	};
	REQUIRE(prog.bpftime_prog_register_raw_helper(std::move(helper)) == 0);
	REQUIRE(prog.bpftime_prog_load(true) == 0);

#ifdef __linux__
	cpu_set_t allowed;
	CPU_ZERO(&allowed);
	REQUIRE(sched_getaffinity(0, sizeof(allowed), &allowed) == 0);
	int pinned_cpu = 0;
	while (pinned_cpu < CPU_SETSIZE && !CPU_ISSET(pinned_cpu, &allowed))
		++pinned_cpu;
	REQUIRE(pinned_cpu < CPU_SETSIZE);
#endif

	std::barrier start(workers + 1);
	std::vector<std::thread> threads;
	for (int t = 0; t < workers; ++t) {
		threads.emplace_back([&, t] {
#ifdef __linux__
			cpu_set_t affinity;
			CPU_ZERO(&affinity);
			CPU_SET(pinned_cpu, &affinity);
			(void)sched_setaffinity(0, sizeof(affinity), &affinity);
#endif
			start.arrive_and_wait();
			for (int iter = 0; iter < 16; ++iter) {
				uint32_t ctx = static_cast<uint32_t>(
					0x10000U + t * 256 + iter);
				uint64_t ret = 0;
				REQUIRE(prog.bpftime_prog_exec(
						&ctx, sizeof(ctx), &ret) == 0);
			}
		});
	}
	start.arrive_and_wait();
	for (auto &th : threads)
		th.join();

	REQUIRE(g_scratch_corruptions.load() == 0);
	REQUIRE(g_scratch_completed.load() ==
		static_cast<uint64_t>(workers * 16));
}

#ifdef __linux__
TEST_CASE("vCPU slot recovers after holder process dies mid-BPF execution",
	  "[percpu][execution]")
{
	const std::string shm_name =
		"bpftime_vcpu_dead_owner_" + std::to_string(getpid());
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

	auto &shm = shm_holder.global_shared_memory;
	pid_t pid = fork();
	REQUIRE(pid >= 0);
	if (pid == 0) {
		// Die while holding CPU 0's slot.
		if (shm.try_lease_vcpu(0) != 0)
			_exit(2);
		raise(SIGKILL);
		_exit(0);
	}
	int status = 0;
	REQUIRE(waitpid(pid, &status, 0) == pid);
	REQUIRE(WIFSIGNALED(status));

	const int real = bpftime_possible_cpus(), slots = shm.vcpu_count();
	REQUIRE(slots == bpftime_vcpu_slots());
	REQUIRE(slots > real);
	// CPU 0's slot recovers from the dead holder; while it is held, CPU 0
	// gets only the extra slots, then nothing.
	std::vector<int> leased{ shm.try_lease_vcpu(0) };
	REQUIRE(leased[0] == 0);
	for (int i = real; i < slots; ++i) {
		const int s = shm.try_lease_vcpu(0);
		REQUIRE((s >= real && s < slots));
		leased.push_back(s);
	}
	uint64_t skips_before = get_shm_trylock_skip_count();
	REQUIRE(shm.try_lease_vcpu(0) == -1);
	REQUIRE(get_shm_trylock_skip_count() == skips_before + 1);
	// Other CPUs' slots were never borrowed.
	if (real > 1) {
		REQUIRE(shm.try_lease_vcpu(1) == 1);
		leased.push_back(1);
	}
	for (int s : leased)
		shm.release_vcpu(s);

	// A thread whose own slot is taken runs on an extra slot, and its
	// bpf_get_smp_processor_id() still reports the CPU it runs on.
	cpu_set_t saved, one;
	REQUIRE(sched_getaffinity(0, sizeof(saved), &saved) == 0);
	const int cpu = sched_getcpu();
	CPU_ZERO(&one);
	CPU_SET(cpu, &one);
	REQUIRE(sched_setaffinity(0, sizeof(one), &one) == 0);
	REQUIRE(shm.try_lease_vcpu(cpu) == cpu);
	REQUIRE(bpftime_enter_bpf_exec());
	REQUIRE(my_sched_getcpu() >= real);
	REQUIRE(bpftime_get_current_cpu() == cpu);
	bpftime_leave_bpf_exec();
	shm.release_vcpu(cpu);
	REQUIRE(sched_setaffinity(0, sizeof(saved), &saved) == 0);
}
#endif
