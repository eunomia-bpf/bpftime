// SPDX-License-Identifier: MIT
//
// Copyright (c) 2026, bpftime contributors
// All rights reserved.
#include "catch2/catch_test_macros.hpp"
#include "bpftime_handler.hpp"
#include "bpftime_shm.hpp"
#include "linux/bpf.h"
#ifndef BPF_F_NO_PREALLOC
#define BPF_F_NO_PREALLOC (1U << 0)
#endif
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace boost::interprocess;
using namespace bpftime;

namespace
{
constexpr int SRC_FD = 1;
constexpr int DUP_FD = 7;

struct lru_key {
	uint32_t thread;
	uint32_t slot;
};

struct lpm_key {
	uint32_t prefixlen;
	uint32_t addr;
};

struct dup_lock_fixture {
	const std::string shm_name;
	shm_remove remover;
	managed_shared_memory segment;
	handler_manager *mgr;
	const boost::interprocess::managed_shared_memory::size_type free_memory;
	dup_lock_fixture()
		: shm_name("DupMapSharedLock-" + std::to_string(getpid())),
		  remover(shm_name.c_str()),
		  segment(create_only, shm_name.c_str(), 8 << 20),
		  free_memory(segment.get_free_memory())
	{
		mgr = segment.construct<handler_manager>(
			"dup_map_lock_manager")(segment, MIN_MAX_FD_COUNT);
	}
};
} // namespace

// A map type whose handler takes map_lock, so a holder of the lock reached
// through one descriptor is observable through the other. LRU_HASH keeps
// should_lock = true; LPM_TRIE, HASH and the map-in-map types do not after
// #1024 and #1027, and lock inside their own impls instead.
TEST_CASE("dup'd LRU hash descriptors share one map lock",
	  "[maps][map_lock][dup]")
{
	dup_lock_fixture fx;
	bpf_map_attr attr{};
	attr.type = static_cast<int>(bpftime::bpf_map_type::BPF_MAP_TYPE_LRU_HASH);
	attr.key_size = sizeof(lru_key);
	attr.value_size = sizeof(uint32_t);
	attr.max_ents = 256;
	REQUIRE(fx.mgr->set_handler(
			SRC_FD,
			bpf_map_handler(SRC_FD, "lru_dup_lock", fx.segment, attr),
			fx.segment) == SRC_FD);
	auto &source = std::get<bpf_map_handler>((*fx.mgr)[SRC_FD]);

	bpf_map_handler duplicate(DUP_FD, source.name.c_str(), fx.segment,
				  source.attr);
	duplicate.share_map_impl_from(source);
	source.inc_map_refcount();
	REQUIRE(fx.mgr->set_handler(DUP_FD, std::move(duplicate),
				    fx.segment) == DUP_FD);
	auto &dup = std::get<bpf_map_handler>((*fx.mgr)[DUP_FD]);

	// Both descriptors must resolve to the same SHM mutex instance.
	REQUIRE(&source.get_map_lock() == &dup.get_map_lock());

	// A holder reached through one descriptor blocks the other descriptor
	// on syscall access and returns -EBUSY on non-blocking helper access.
	std::atomic<bool> waiter_entered{ false };
	REQUIRE(source.get_map_lock().lock("dup-source"));
	lru_key busy_key{ 0, 0 };
	uint32_t busy_val = 1;
	REQUIRE(dup.map_update_elem(&busy_key, &busy_val, BPF_ANY, false) ==
		-EBUSY);
	std::thread waiter([&] {
		REQUIRE(dup.get_map_lock().lock("dup-waiter"));
		waiter_entered.store(true, std::memory_order_release);
		dup.get_map_lock().unlock();
	});
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	REQUIRE_FALSE(waiter_entered.load(std::memory_order_acquire));
	source.get_map_lock().unlock();
	waiter.join();
	REQUIRE(waiter_entered.load(std::memory_order_acquire));

	// Concurrent syscall updates/lookups across both dup'd descriptors
	// serialize on the shared SHM lock and stay coherent.
	constexpr int kThreads = 4;
	constexpr int kIters = 5000;
	constexpr int kKeysPerThread = 16;
	std::atomic<bool> go{ false };
	std::atomic<int> errors{ 0 };
	std::vector<std::thread> workers;
	for (int t = 0; t < kThreads; ++t) {
		workers.emplace_back([&, t] {
			const bpf_map_handler &writer =
				(t % 2 == 0) ? source : dup;
			const bpf_map_handler &reader =
				(t % 2 == 0) ? dup : source;
			while (!go.load(std::memory_order_acquire))
				std::this_thread::yield();
			for (int i = 0; i < kIters; ++i) {
				lru_key key{ static_cast<uint32_t>(t),
					     static_cast<uint32_t>(
						     i % kKeysPerThread) };
				uint32_t value = static_cast<uint32_t>(t + i);
				if (writer.map_update_elem(&key, &value,
							   BPF_ANY,
							   true) != 0) {
					errors.fetch_add(1,
							 std::memory_order_relaxed);
					return;
				}
				const uint32_t *got =
					static_cast<const uint32_t *>(
						reader.map_lookup_elem(&key,
								       true));
				if (got == nullptr || *got != value) {
					errors.fetch_add(1,
							 std::memory_order_relaxed);
					return;
				}
			}
		});
	}
	go.store(true, std::memory_order_release);
	for (auto &th : workers)
		th.join();
	REQUIRE(errors.load(std::memory_order_relaxed) == 0);

	fx.mgr->clear_id_at(DUP_FD, fx.segment);
	fx.mgr->clear_id_at(SRC_FD, fx.segment);
	fx.segment.destroy_ptr(fx.mgr);
	REQUIRE(fx.segment.get_free_memory() == fx.free_memory);
}

// LPM_TRIE takes no map_lock on either path (#1024); what a dup'd descriptor
// still has to share is the mutex identity, and its concurrent syscall
// updates and lookups have to stay coherent through both descriptors.
TEST_CASE("dup'd LPM trie descriptors share one map lock",
	  "[maps][map_lock][dup]")
{
	dup_lock_fixture fx;
	bpf_map_attr attr{};
	attr.type = static_cast<int>(bpftime::bpf_map_type::BPF_MAP_TYPE_LPM_TRIE);
	attr.key_size = sizeof(lpm_key);
	attr.value_size = sizeof(uint32_t);
	attr.max_ents = 256;
	attr.flags = BPF_F_NO_PREALLOC;
	REQUIRE(fx.mgr->set_handler(
			SRC_FD,
			bpf_map_handler(SRC_FD, "lpm_dup_lock", fx.segment, attr),
			fx.segment) == SRC_FD);
	auto &source = std::get<bpf_map_handler>((*fx.mgr)[SRC_FD]);

	bpf_map_handler duplicate(DUP_FD, source.name.c_str(), fx.segment,
				  source.attr);
	duplicate.share_map_impl_from(source);
	source.inc_map_refcount();
	REQUIRE(fx.mgr->set_handler(DUP_FD, std::move(duplicate),
				    fx.segment) == DUP_FD);
	auto &dup = std::get<bpf_map_handler>((*fx.mgr)[DUP_FD]);

	REQUIRE(&source.get_map_lock() == &dup.get_map_lock());

	constexpr int kThreads = 4;
	constexpr int kIters = 5000;
	constexpr int kKeysPerThread = 16;
	std::atomic<bool> go{ false };
	std::atomic<int> errors{ 0 };
	std::vector<std::thread> workers;
	for (int t = 0; t < kThreads; ++t) {
		workers.emplace_back([&, t] {
			const bpf_map_handler &writer =
				(t % 2 == 0) ? source : dup;
			const bpf_map_handler &reader =
				(t % 2 == 0) ? dup : source;
			while (!go.load(std::memory_order_acquire))
				std::this_thread::yield();
			for (int i = 0; i < kIters; ++i) {
				lpm_key key{ 32,
					     static_cast<uint32_t>(
						     t * kKeysPerThread +
						     i % kKeysPerThread) };
				uint32_t value = static_cast<uint32_t>(t + i);
				if (writer.map_update_elem(&key, &value,
							   BPF_ANY,
							   true) != 0) {
					errors.fetch_add(1,
							 std::memory_order_relaxed);
					return;
				}
				const uint32_t *got =
					static_cast<const uint32_t *>(
						reader.map_lookup_elem(&key,
								       true));
				if (got == nullptr || *got != value) {
					errors.fetch_add(1,
							 std::memory_order_relaxed);
					return;
				}
			}
		});
	}
	go.store(true, std::memory_order_release);
	for (auto &th : workers)
		th.join();
	REQUIRE(errors.load(std::memory_order_relaxed) == 0);

	fx.mgr->clear_id_at(DUP_FD, fx.segment);
	fx.mgr->clear_id_at(SRC_FD, fx.segment);
	fx.segment.destroy_ptr(fx.mgr);
	REQUIRE(fx.segment.get_free_memory() == fx.free_memory);
}
