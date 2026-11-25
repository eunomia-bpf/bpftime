/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#include "catch2/catch_test_macros.hpp"

#include "handler/perf_event_handler.hpp"
#include "common_def.hpp"

#include <atomic>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#if __linux__
#include <sys/syscall.h>
#include <sys/wait.h>
#endif

#if __linux__

namespace
{
struct cache_payload {
	int event_index;
	int sequence;
};

// Owns one shared-memory segment holding more distinct software perf events
// than the per-thread shard cache can hold (the cache capacity is 6).
struct cache_fixture {
	static constexpr size_t event_count = 8;
	static constexpr size_t ring_size = 64 * 1024;

	const std::string shm_name;
	shm_remove remover;
	boost::interprocess::managed_shared_memory shm;
	std::vector<bpftime::software_perf_event_data *> events;

	cache_fixture()
		: shm_name("SoftwarePerfCacheTestShm-" +
			   std::to_string(getpid())),
		  remover(std::string(shm_name)),
		  shm(boost::interprocess::create_only, shm_name.c_str(),
		      16 * 1024 * 1024)
	{
		events.reserve(event_count);
		for (size_t i = 0; i < event_count; ++i) {
			auto *event =
				shm.construct<bpftime::software_perf_event_data>(
					("perf" + std::to_string(i)).c_str())(
					0, 0, 0, shm);
			REQUIRE(event != nullptr);
			REQUIRE(event->ensure_mmap_buffer(
					getpagesize() + ring_size) != nullptr);
			events.push_back(event);
		}
	}

	int output(bpftime::software_perf_event_data *event, int index,
		   int sequence)
	{
		cache_payload payload{ index, sequence };
		return event->output_data(&payload, sizeof(payload));
	}
};
} // namespace

TEST_CASE("Software perf shard cache evicts and rebuilds beyond its capacity",
	  "[perf_event][software_perf_event][cache]")
{
	cache_fixture fixture;

	// Touching more distinct events than cache slots forces the bounded
	// eviction path (round-robin replacement of one occupied slot) on one
	// thread. Every output must still succeed and remain readable.
	for (size_t i = 0; i < cache_fixture::event_count; ++i) {
		REQUIRE(fixture.output(fixture.events[i],
				       static_cast<int>(i), 0) == 0);
	}
	for (size_t i = 0; i < cache_fixture::event_count; ++i) {
		REQUIRE(fixture.events[i]->has_data());
	}

	// After eviction, an early event has no live cache entry. Output again
	// and verify the entry is reinstalled instead of corrupting another
	// event's slot.
	REQUIRE(fixture.output(fixture.events[0], 0, 1) == 0);
	REQUIRE(fixture.output(fixture.events[1], 1, 1) == 0);
	REQUIRE(fixture.events[0]->has_data());
	REQUIRE(fixture.events[1]->has_data());
}

TEST_CASE("Software perf shard cache stays independent across threads",
	  "[perf_event][software_perf_event][cache]")
{
	cache_fixture fixture;
	std::atomic<bool> start{ false };
	std::atomic<bool> failed{ false };

	auto worker = [&](int worker_id) {
		while (!start.load(std::memory_order_acquire))
			std::this_thread::yield();
		// Each thread walks every event, so both threads fill and
		// evict their own thread-local cache concurrently.
		for (size_t round = 0; round < 2; ++round) {
			for (size_t i = 0; i < cache_fixture::event_count;
			     ++i) {
				if (fixture.output(fixture.events[i], worker_id,
						   static_cast<int>(round)) !=
				    0) {
					failed.store(true,
						     std::memory_order_release);
				}
			}
		}
	};

	std::thread first(worker, 0);
	std::thread second(worker, 1);
	start.store(true, std::memory_order_release);
	first.join();
	second.join();

	REQUIRE_FALSE(failed.load(std::memory_order_acquire));
	for (auto *event : fixture.events) {
		REQUIRE(event->has_data());
	}
}


TEST_CASE("Software perf shard cache replaces one slot instead of clearing",
	  "[perf_event][software_perf_event][cache]")
{
	cache_fixture fixture;
	constexpr size_t slots = 6;
	REQUIRE(cache_fixture::event_count > slots);

	for (size_t i = 0; i < slots; ++i)
		REQUIRE(fixture.output(fixture.events[i],
				       static_cast<int>(i), 0) == 0);
	REQUIRE(bpftime::cached_software_perf_event_shard_count() == slots);

	// One extra event no longer empties the table.
	REQUIRE(fixture.output(fixture.events[slots],
			       static_cast<int>(slots), 0) == 0);
	REQUIRE(bpftime::cached_software_perf_event_shard_count() == slots);

	// Fill the rest; the table stays full rather than collapsing to one slot.
	for (size_t i = slots + 1; i < cache_fixture::event_count; ++i)
		REQUIRE(fixture.output(fixture.events[i],
				       static_cast<int>(i), 0) == 0);
	REQUIRE(bpftime::cached_software_perf_event_shard_count() == slots);
}

TEST_CASE("Software perf shard resolves the child's tid after fork",
	  "[perf_event][software_perf_event][cache]")
{
	const std::string shm_name =
		"SoftwarePerfForkTestShm-" + std::to_string(getpid());
	shm_remove remover{ std::string(shm_name) };
	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, shm_name.c_str(),
		16 * 1024 * 1024);
	auto *event = shm.construct<bpftime::software_perf_event_data>(
		"fork_perf")(0, 0, 0, shm);
	REQUIRE(event != nullptr);
	REQUIRE(event->ensure_mmap_buffer(getpagesize() + 1024 * 1024) !=
		nullptr);

	// Populate the parent's thread-local tid cache and its shard first.
	int parent_payload = 1;
	REQUIRE(event->output_data(&parent_payload,
				   sizeof(parent_payload)) == 0);

	const pid_t pid = fork();
	REQUIRE(pid >= 0);
	if (pid == 0) {
		// _exit skips destructors: the shared memory is owned and
		// cleaned up by the parent. Without the atfork reset the
		// inherited TLS cache would make this output reuse the
		// parent's tid.
		int child_payload = 2;
		if (event->output_data(&child_payload,
				       sizeof(child_payload)) != 0)
			_exit(3);
		const int64_t child_tid =
			static_cast<int64_t>(syscall(SYS_gettid));
		bool found_child_shard = false;
		for (const auto &shard : event->producer_shards) {
			if (shard.tid == child_tid)
				found_child_shard = true;
		}
		_exit(found_child_shard ? 0 : 4);
	}

	int status = 0;
	REQUIRE(waitpid(pid, &status, 0) == pid);
	REQUIRE(WIFEXITED(status));
	// 3 = output failed in the child; 4 = the child kept the parent's tid.
	REQUIRE(WEXITSTATUS(status) == 0);
}

#endif
