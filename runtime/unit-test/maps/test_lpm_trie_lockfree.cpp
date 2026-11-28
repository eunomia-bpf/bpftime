/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, bpftime contributors
 * All rights reserved.
 */
#include "catch2/catch_test_macros.hpp"
#include "bpf_map/userspace/lpm_trie_map.hpp"
#include "bpftime_handler.hpp"
#include "bpftime_shm.hpp"
#include "robust_shm_mutex.hpp"
#include "../common_def.hpp"
#include "linux/bpf.h"
#include <boost/interprocess/managed_shared_memory.hpp>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <chrono>
#include <cstring>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

using bpftime::lpm_trie_map_impl;

namespace
{
struct ipv4_key {
	uint32_t prefixlen;
	uint8_t ip[4];
};

ipv4_key k4(uint8_t a, uint8_t b, uint8_t c, uint8_t d, uint32_t plen)
{
	return ipv4_key{ plen, { a, b, c, d } };
}

bool one_of(const void *v, const std::vector<uint32_t> &allowed)
{
	if (v == nullptr)
		return false;
	uint32_t got;
	std::memcpy(&got, v, sizeof(got));
	for (uint32_t a : allowed)
		if (got == a)
			return true;
	return false;
}
} // namespace

TEST_CASE("LPM trie lookups never fail while a writer inserts and deletes",
	  "[lpm_trie][lockfree]")
{
	const std::string name = "LPMLockFreeChurn-" + std::to_string(getpid());
	shm_remove remover{ std::string(name) };
	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, name.c_str(), 4 << 20);
	// 1024 entries: the 2048-node pool keeps about 2000 freed nodes queued,
	// so a node is recycled at least about 1000 writes after its free.
	auto *trie = shm.construct<lpm_trie_map_impl>("lpm")(
		shm, sizeof(ipv4_key), sizeof(uint32_t), 1024);
	REQUIRE(trie != nullptr);

	auto put = [&](ipv4_key k, uint32_t v) {
		return trie->elem_update(&k, &v, BPF_ANY);
	};
	REQUIRE(put(k4(10, 0, 0, 0, 8), 8) == 0);
	REQUIRE(put(k4(10, 1, 0, 0, 16), 16) == 0);

	// Churned prefixes sit on the probes' paths, so readers walk through
	// nodes that are being linked, unlinked and recycled.
	const std::vector<std::pair<ipv4_key, uint32_t> > churn = {
		{ k4(10, 1, 5, 0, 24), 24 },   { k4(10, 1, 5, 0, 28), 28 },
		{ k4(10, 1, 5, 7, 32), 32 },   { k4(10, 1, 4, 0, 24), 1024 },
		{ k4(10, 1, 128, 0, 17), 17 },
	};
	struct probe {
		ipv4_key key;
		std::vector<uint32_t> allowed;
	};
	const probe probes[] = {
		{ k4(10, 1, 5, 7, 32), { 16, 24, 28, 32 } },
		{ k4(10, 1, 4, 9, 32), { 16, 1024 } },
		{ k4(10, 3, 0, 1, 32), { 8 } },
	};

	const uint64_t skips_before = bpftime::get_shm_trylock_skip_count();
	std::atomic<bool> stop{ false };
	std::atomic<uint64_t> lookups{ 0 }, nulls{ 0 }, wrong{ 0 }, stale{ 0 },
		write_errors{ 0 };
	std::vector<std::thread> readers;
	for (int t = 0; t < 4; ++t)
		readers.emplace_back([&]() {
			while (!stop.load(std::memory_order_relaxed)) {
				for (const probe &p : probes) {
					const auto t0 = std::chrono::steady_clock::now();
					const void *v = trie->elem_lookup(&p.key);
					if (v == nullptr) {
						nulls.fetch_add(1);
						continue;
					}
					if (one_of(v, p.allowed))
						continue;
					// Another key's value means the node was
					// recycled after the lookup returned. That needs
					// the free queue to cycle, so it is only legitimate
					// for a reader stalled far longer than a lookup.
					if (!one_of(v, { 8, 16, 17, 24, 28, 32, 1024 }) ||
					    std::chrono::steady_clock::now() - t0 <
						    std::chrono::microseconds(20))
						wrong.fetch_add(1);
					else
						stale.fetch_add(1);
				}
				lookups.fetch_add(3, std::memory_order_relaxed);
			}
		});

	for (int round = 0; round < 20000; ++round) {
		for (auto &[k, v] : churn)
			if (put(k, v) != 0)
				write_errors.fetch_add(1);
		for (size_t i = 0; i < churn.size(); ++i) {
			auto k = churn[round % 2 ? i : churn.size() - 1 - i].first;
			if (trie->elem_delete(&k) != 0)
				write_errors.fetch_add(1);
		}
	}
	stop.store(true);
	for (auto &t : readers)
		t.join();

	INFO("lookups=" << lookups.load() << " nulls=" << nulls.load()
			<< " wrong=" << wrong.load() << " stale=" << stale.load()
			<< " write_errors=" << write_errors.load());
	REQUIRE(lookups.load() > 0);
	REQUIRE(write_errors.load() == 0);
	REQUIRE(nulls.load() == 0);
	REQUIRE(wrong.load() == 0);
	REQUIRE(bpftime::get_shm_trylock_skip_count() == skips_before);
	REQUIRE(trie->get_current_entries() == 2);
	REQUIRE(one_of(trie->elem_lookup(&probes[0].key), { 16 }));
	shm.destroy_ptr(trie);
}

TEST_CASE("LPM trie stays usable after a writer dies mid-update",
	  "[lpm_trie][lockfree][dead_owner]")
{
	const std::string name = "LPMLockFreeDeadWriter-" + std::to_string(getpid());
	shm_remove remover{ std::string(name) };
	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, name.c_str(), 1 << 20);
	const long page_sz = sysconf(_SC_PAGESIZE);
	REQUIRE(page_sz >= 4096);
	const size_t val_sz = static_cast<size_t>(page_sz) * 2;
	constexpr unsigned MAX = 8;
	auto *trie = shm.construct<lpm_trie_map_impl>("lpm")(
		shm, sizeof(ipv4_key), val_sz, MAX);
	REQUIRE(trie != nullptr);
	std::vector<uint8_t> good(val_sz, 0x5a);
	auto base = k4(10, 0, 0, 0, 8);
	REQUIRE(trie->elem_update(&base, good.data(), BPF_ANY) == 0);

	// The child dies with SIGSEGV while copying the value of a new key,
	// holding the write lock, after taking a node but before linking it.
	pid_t child = fork();
	REQUIRE(child >= 0);
	if (child == 0) {
		std::signal(SIGSEGV, SIG_DFL);
		void *p = mmap(nullptr, val_sz, PROT_READ | PROT_WRITE,
			       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		if (p == MAP_FAILED ||
		    mprotect(static_cast<uint8_t *>(p) + page_sz, page_sz,
			     PROT_NONE) != 0)
			_exit(2);
		auto k = k4(10, 1, 0, 0, 16);
		(void)trie->elem_update(&k, p, BPF_ANY);
		_exit(0);
	}
	int status = 0;
	REQUIRE(waitpid(child, &status, 0) == child);
	REQUIRE_FALSE((WIFEXITED(status) && WEXITSTATUS(status) == 0));

	// The tree is intact and the lock is recovered, not poisoned: the
	// helper (try_lock) path writes at once.
	auto k16 = k4(10, 1, 0, 0, 16);
	auto probe = k4(10, 1, 2, 3, 32);
	REQUIRE(trie->get_current_entries() == 1);
	REQUIRE(trie->elem_lookup(&k16) != nullptr); // falls back to the /8
	REQUIRE(trie->elem_update(&k16, good.data(), BPF_ANY, false) == 0);
	REQUIRE(trie->elem_lookup(&probe) != nullptr);
	REQUIRE(trie->elem_delete(&k16, false) == 0);
	REQUIRE(trie->elem_delete(&base, true) == 0);
	REQUIRE(trie->get_current_entries() == 0);

	// The one node the dead writer took is leaked. MAX host routes need
	// 2 * MAX - 1 nodes, which the remaining pool still holds.
	for (uint8_t i = 0; i < MAX; ++i) {
		auto k = k4(10, 2, 0, i, 32);
		REQUIRE(trie->elem_update(&k, good.data(), BPF_NOEXIST) == 0);
	}
	auto extra = k4(10, 3, 0, 0, 32);
	errno = 0;
	REQUIRE(trie->elem_update(&extra, good.data(), BPF_ANY) == -1);
	REQUIRE(errno == ENOSPC);
	for (uint8_t i = 0; i < MAX; ++i) {
		auto k = k4(10, 2, 0, i, 32);
		REQUIRE(trie->elem_lookup(&k) != nullptr);
	}
	shm.destroy_ptr(trie);
}

TEST_CASE("LPM trie helper lookups from many threads of one process never skip",
	  "[lpm_trie][lockfree]")
{
	const std::string name = "LPMLockFreeSamePid-" + std::to_string(getpid());
	shm_remove remover{ std::string(name) };
	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, name.c_str(), 4 << 20);
	constexpr int MAP_FD = 1;
	auto *mgr = shm.construct<bpftime::handler_manager>("LPMSamePidMgr")(
		shm, MIN_MAX_FD_COUNT);
	REQUIRE(mgr != nullptr);
	REQUIRE(mgr->set_handler(
			MAP_FD,
			bpftime::bpf_map_handler(
				MAP_FD,
				(int)bpftime::bpf_map_type::BPF_MAP_TYPE_LPM_TRIE,
				sizeof(ipv4_key), sizeof(uint32_t), 100,
				BPF_F_NO_PREALLOC, "lpm_same_pid", shm),
			shm) == MAP_FD);
	auto &map = std::get<bpftime::bpf_map_handler>((*mgr)[MAP_FD]);
	REQUIRE(const_cast<bpftime::bpf_map_handler &>(map).map_init(shm) == 0);

	auto k16 = k4(192, 168, 0, 0, 16), k24 = k4(192, 168, 1, 0, 24);
	uint32_t one = 1, two = 2;
	REQUIRE(map.map_update_elem(&k16, &one, 0, true) == 0);
	REQUIRE(map.map_update_elem(&k24, &two, 0, true) == 0);
	const auto probe = k4(192, 168, 1, 7, 32);

	// Helper-path lookups used to trylock map_lock and returned NULL
	// whenever another thread of the process held it.
	const uint64_t skips_before = bpftime::get_shm_trylock_skip_count();
	std::atomic<bool> stop{ false };
	std::atomic<uint64_t> failures{ 0 };
	std::thread writer([&]() {
		while (!stop.load(std::memory_order_relaxed))
			if (map.map_update_elem(&k24, &two, BPF_EXIST, false) != 0)
				failures.fetch_add(1);
	});
	std::vector<std::thread> readers;
	for (int t = 0; t < 8; ++t)
		readers.emplace_back([&]() {
			for (int i = 0; i < 50000; ++i)
				if (!one_of(map.map_lookup_elem(&probe, false),
					    { 2 }))
					failures.fetch_add(1);
		});
	for (auto &t : readers)
		t.join();
	stop.store(true);
	writer.join();

	REQUIRE(failures.load() == 0);
	REQUIRE(bpftime::get_shm_trylock_skip_count() == skips_before);
	mgr->clear_all(shm);
	shm.destroy_ptr(mgr);
}
