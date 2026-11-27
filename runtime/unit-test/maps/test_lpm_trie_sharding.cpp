// SPDX-License-Identifier: MIT
// Regression coverage for the 64-way LPM_TRIE sharding change:
//  - mixed short (<32-bit, fallback stripe) and long (hashed stripe) prefixes
//    resolve with correct longest-prefix precedence via the nested lookup;
//  - a reserved entry slot is rolled back when node allocation fails, so
//    n_entries never counts a node that did not commit;
//  - helper (from_syscall=false) lookups preserve host errno and never return
//    a torn value while another thread deletes and re-inserts the same key.
#include "catch2/catch_test_macros.hpp"

#include "bpf_map/userspace/lpm_trie_map.hpp"
#include "../common_def.hpp"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace bpftime;

namespace
{
// {prefixlen, pid (big endian), addr}: 12 key-data bytes, prefixlen 96 covers
// all of them; a prefixlen below 32 lands in the fallback stripe.
struct lpm_pid_key {
	uint32_t prefixlen;
	uint32_t pid_be;
	uint64_t addr;
};

lpm_pid_key make_long_key(uint32_t pid, uint64_t addr)
{
	return lpm_pid_key{ 96, __builtin_bswap32(pid), addr };
}
} // namespace

TEST_CASE("LPM sharding mixes fallback and hashed stripes with LPM precedence",
	  "[lpm_trie][lpm_trie_map][sharding]")
{
	const std::string shm_name =
		"LpmShardingMixedShm-" + std::to_string(getpid());
	shm_remove remover(std::move(shm_name));
	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, shm_name.c_str(),
		size_t(1) * 1024 * 1024);

	auto *trie = shm.construct<lpm_trie_map_impl>("mixed")(
		shm, sizeof(lpm_pid_key), sizeof(uint64_t), 128);
	REQUIRE(trie != nullptr);

	// A short /16 prefix lives only in the fallback stripe.
	lpm_pid_key short_key{ 16, __builtin_bswap32(0xABCD0000), 0 };
	uint64_t short_val = 11;
	REQUIRE(trie->elem_update(&short_key, &short_val, 0) == 0);

	// A fully-specified /96 key lives in a hashed long-prefix stripe.
	lpm_pid_key long_key = make_long_key(0x12345678, 0x5000);
	uint64_t long_val = 22;
	REQUIRE(trie->elem_update(&long_key, &long_val, 0) == 0);
	REQUIRE(trie->get_current_entries() == 2);

	// Exact long match wins from its own stripe.
	void *got = trie->elem_lookup(&long_key, true);
	REQUIRE(got != nullptr);
	REQUIRE(*(uint64_t *)got == long_val);

	// A /96 query with no exact long entry but whose leading 16 bits match
	// the short prefix must fall through to the fallback stripe and return
	// the short value, not a false miss.
	lpm_pid_key fallback_query = make_long_key(0xABCD9999, 0x1);
	got = trie->elem_lookup(&fallback_query, true);
	REQUIRE(got != nullptr);
	REQUIRE(*(uint64_t *)got == short_val);

	// An unrelated /96 query matches neither stripe.
	lpm_pid_key miss_query = make_long_key(0x01020304, 0x2);
	errno = 0;
	REQUIRE(trie->elem_lookup(&miss_query, true) == nullptr);
	REQUIRE(errno == ENOENT);

	shm.destroy_ptr(trie);
}

TEST_CASE("LPM sharding rolls back the reserved slot when allocation fails",
	  "[lpm_trie][lpm_trie_map][sharding]")
{
	const std::string shm_name =
		"LpmShardingRollbackShm-" + std::to_string(getpid());
	shm_remove remover(std::move(shm_name));
	// Deliberately small so node storage exhausts long before max_entries,
	// while still fitting the 65-mutex map object and segment metadata.
	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, shm_name.c_str(),
		size_t(64) * 1024);

	struct key4 {
		uint32_t prefixlen;
		uint32_t addr_be;
	};
	auto *trie = shm.construct<lpm_trie_map_impl>("rollback")(
		shm, sizeof(key4), sizeof(uint64_t), 100000);
	REQUIRE(trie != nullptr);

	long committed = 0;
	bool failed = false;
	for (uint32_t i = 1; i < 5000 && !failed; ++i) {
		key4 k{ 32, __builtin_bswap32(i) };
		uint64_t v = i;
		try {
			long rc = trie->elem_update(&k, &v, 0);
			if (rc == 0) {
				++committed;
			} else {
				REQUIRE(errno == ENOMEM || errno == ENOSPC);
				failed = true;
			}
		} catch (const boost::interprocess::bad_alloc &) {
			failed = true;
		}
	}
	REQUIRE(failed);
	REQUIRE(committed > 0);

	// The failed reservation must not remain counted: the counter equals
	// exactly the real, retrievable entries rather than committed + leaked.
	REQUIRE(trie->get_current_entries() == static_cast<size_t>(committed));
	for (uint32_t i = 1; i <= static_cast<uint32_t>(committed); ++i) {
		key4 k{ 32, __builtin_bswap32(i) };
		void *got = trie->elem_lookup(&k, true);
		REQUIRE(got != nullptr);
		REQUIRE(*(uint64_t *)got == i);
	}

	shm.destroy_ptr(trie);
}

TEST_CASE("LPM helper lookups preserve errno under concurrent delete",
	  "[lpm_trie][lpm_trie_map][sharding]")
{
	const std::string shm_name =
		"LpmShardingErrnoShm-" + std::to_string(getpid());
	shm_remove remover(std::move(shm_name));
	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, shm_name.c_str(),
		size_t(4) * 1024 * 1024);

	auto *trie = shm.construct<lpm_trie_map_impl>("errno")(
		shm, sizeof(lpm_pid_key), sizeof(uint64_t), 4096);
	REQUIRE(trie != nullptr);

	const lpm_pid_key flap_key = make_long_key(2000, 0x7000);
	constexpr uint64_t flap_val = 99;
	REQUIRE(trie->elem_update(&flap_key, &flap_val, 0, true) == 0);

	std::atomic<bool> stop{ false };
	std::atomic<uint64_t> bad_value{ 0 };
	std::atomic<uint64_t> errno_dirty{ 0 };
	constexpr int kSentinelErrno = 12345;

	std::vector<std::thread> readers;
	for (int t = 0; t < 8; ++t)
		readers.emplace_back([&] {
			while (!stop.load(std::memory_order_acquire)) {
				errno = kSentinelErrno;
				// Helper path: non-blocking trylock, must keep
				// host errno whether it hits, misses, or skips.
				void *got = trie->elem_lookup(&flap_key, false);
				if (errno != kSentinelErrno)
					errno_dirty.fetch_add(
						1, std::memory_order_relaxed);
				if (got && *(uint64_t *)got != flap_val)
					bad_value.fetch_add(
						1, std::memory_order_relaxed);
			}
		});

	std::thread flapper([&] {
		uint64_t v = flap_val;
		for (int i = 0; i < 200000; ++i) {
			// Best-effort helper delete/reinsert to overlap lookups
			// with the is_intermediate transition; trylock skips are
			// acceptable because presence is re-established below.
			(void)trie->elem_delete(&flap_key, false);
			(void)trie->elem_update(&flap_key, &v, 0, false);
		}
		stop.store(true, std::memory_order_release);
	});

	flapper.join();
	for (auto &th : readers)
		th.join();

	REQUIRE(bad_value.load() == 0);
	REQUIRE(errno_dirty.load() == 0);
	// Blocking control-plane write guarantees the key is real regardless of
	// any helper trylock skips during the storm.
	REQUIRE(trie->elem_update(&flap_key, &flap_val, 0, true) == 0);
	void *final_got = trie->elem_lookup(&flap_key, true);
	REQUIRE(final_got != nullptr);
	REQUIRE(*(uint64_t *)final_got == flap_val);

	shm.destroy_ptr(trie);
}
