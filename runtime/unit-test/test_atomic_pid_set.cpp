/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#include "catch2/catch_test_macros.hpp"
#include "bpftime_shm_internal.hpp"
#include <set>
#include <string>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace
{
std::set<int> snapshot(const bpftime::alive_agent_pids &pids)
{
	std::set<int> out;
	pids.iterate([&](int pid) { out.insert(pid); });
	return out;
}
} // namespace

TEST_CASE("atomic_pid_set keeps idempotent inserts and removals consistent")
{
	bpftime::alive_agent_pids pids;
	REQUIRE(pids.insert(11));
	REQUIRE(pids.insert(11)); // duplicate insert is a no-op
	REQUIRE(pids.insert(22));
	REQUIRE(snapshot(pids) == std::set<int>{ 11, 22 });
	REQUIRE_FALSE(pids.insert(0));
	REQUIRE_FALSE(pids.insert(-1));
	REQUIRE(pids.remove(11));
	REQUIRE_FALSE(pids.remove(11)); // removing twice is a no-op
	REQUIRE(snapshot(pids) == std::set<int>{ 22 });
}

TEST_CASE("atomic_pid_set survives concurrent insert and remove")
{
	bpftime::alive_agent_pids pids;
	constexpr int kThreads = 8, kPerThread = 16;
	std::vector<std::thread> threads;
	for (int t = 0; t < kThreads; ++t) {
		threads.emplace_back([&, t] {
			int base = t * kPerThread + 1;
			for (int round = 0; round < 200; ++round)
				for (int i = 0; i < kPerThread; ++i) {
					pids.insert(base + i);
					pids.remove(base + i);
				}
		});
	}
	for (auto &th : threads)
		th.join();
	REQUIRE(snapshot(pids).empty());
	for (int t = 0; t < kThreads; ++t)
		for (int i = 0; i < kPerThread; ++i)
			REQUIRE(pids.insert(t * kPerThread + 1 + i));
	REQUIRE(snapshot(pids).size() ==
		static_cast<size_t>(kThreads * kPerThread));
}

#if defined(__linux__)
TEST_CASE("atomic_pid_set reclaims dead slots before reporting exhaustion")
{
	using table_t = bpftime::atomic_pid_set<4>;
	table_t pids;
	REQUIRE(pids.insert(getpid()));
	for (int i = 0; i < 3; ++i) {
		pid_t child = fork();
		REQUIRE(child >= 0);
		if (child == 0)
			_exit(0);
		int status;
		REQUIRE(waitpid(child, &status, 0) == child);
		REQUIRE(pids.insert(child)); // stale slot from a dead peer
	}
	// Table is full with three dead PIDs; the live parent PID must be
	// admitted after the dead slots are reclaimed.
	REQUIRE(pids.insert(getppid()));
	std::set<int> left;
	pids.iterate([&](int pid) { left.insert(pid); });
	REQUIRE(left == std::set<int>{ getpid(), getppid() });
}
#endif
#if defined(__linux__)
TEST_CASE("atomic_pid_set stays usable after a peer process is SIGKILLed")
{
	using table_t = bpftime::atomic_pid_set<64>;
	const std::string name =
		"bpftime-pid-set-kill-test-" + std::to_string(getpid());
	boost::interprocess::shared_memory_object::remove(name.c_str());
	{
		boost::interprocess::managed_shared_memory segment(
			boost::interprocess::create_only, name.c_str(),
			1 << 20);
		auto *pids = segment.find_or_construct<table_t>("pids")();
		pid_t child = fork();
		REQUIRE(child >= 0);
		if (child == 0) {
			// Mutate until killed; a lock-free table has no lock
			// state the kill can leak to other processes.
			for (;;) {
				pids->insert(getpid());
				pids->remove(getpid());
			}
			_exit(0);
		}
		usleep(50000);
		REQUIRE(kill(child, SIGKILL) == 0);
		int status;
		REQUIRE(waitpid(child, &status, 0) == child);
		REQUIRE(pids->insert(getpid()));
		REQUIRE(pids->insert(getpid()));
		bool seen = false;
		pids->iterate([&](int pid) {
			if (pid == getpid())
				seen = true;
		});
		REQUIRE(seen);
		REQUIRE(pids->remove(getpid()));
	}
	boost::interprocess::shared_memory_object::remove(name.c_str());
}
#endif
