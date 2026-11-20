/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#include "catch2/catch_test_macros.hpp"

#include "bpf_map/userspace/lpm_trie_map.hpp"
#include "platform_utils.hpp"
#include "../common_def.hpp"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <cstdint>
#include <cstring>
#include <string>
#include <sched.h>
#include <unistd.h>
#include <thread>
#include <vector>

namespace
{
// Boundary between the small inline TLS buffer and the heap TLS fallback on
// the non-BPF (bpf() syscall) path.
constexpr size_t small_value_boundary = 256;
using bpftime::bpf_lpm_trie_key;

// Pins the current thread's BPF CPU for the scope, mirroring
// bpf_cpu_execution_guard which always sets a pinned BPF CPU before a BPF
// program runs. On destruction it restores the "no BPF context" state so the
// non-BPF (syscall) path can be exercised independently.
struct scoped_bpf_cpu {
	int previous;
	explicit scoped_bpf_cpu(int cpu) : previous(bpftime_get_current_bpf_cpu())
	{
		bpftime_set_current_bpf_cpu(cpu);
	}
	~scoped_bpf_cpu()
	{
		bpftime_set_current_bpf_cpu(previous);
	}
	scoped_bpf_cpu(const scoped_bpf_cpu &) = delete;
	scoped_bpf_cpu &operator=(const scoped_bpf_cpu &) = delete;
};

struct lpm_fixture {
	const std::string shm_name;
	shm_remove remover;
	boost::interprocess::managed_shared_memory shm;
	bpftime::lpm_trie_map_impl *trie = nullptr;

	lpm_fixture(size_t value_size)
		: shm_name("LPMInlineValueTestShm-" + std::to_string(getpid()) +
			   "-" + std::to_string(value_size)),
		  remover(std::string(shm_name)),
		  shm(boost::interprocess::create_only, shm_name.c_str(),
		      8 * 1024 * 1024)
	{
		trie = shm.construct<bpftime::lpm_trie_map_impl>(
			"LPMInlineValueInstance")(shm, sizeof(bpf_lpm_trie_key) + 4,
						  value_size, 64);
		REQUIRE(trie != nullptr);
	}

	std::vector<uint8_t> make_key(uint8_t a, uint8_t b, uint8_t c,
				      uint8_t d)
	{
		std::vector<uint8_t> storage(sizeof(bpf_lpm_trie_key) + 4, 0);
		auto *key =
			reinterpret_cast<bpf_lpm_trie_key *>(storage.data());
		key->prefixlen = 32;
		key->data[0] = a;
		key->data[1] = b;
		key->data[2] = c;
		key->data[3] = d;
		return storage;
	}

	bool result_lives_in_shm(const void *ptr) const
	{
		return ptr != nullptr && shm.belongs_to_segment(ptr);
	}

	// Proves the whole value range, not just its first byte, stays inside the
	// SHM segment (a stride change cannot silently overrun the scratch).
	bool result_range_in_shm(const void *ptr, size_t size) const
	{
		if (ptr == nullptr || size == 0)
			return false;
		const auto *begin = static_cast<const uint8_t *>(ptr);
		return shm.belongs_to_segment(begin) &&
		       shm.belongs_to_segment(begin + size - 1);
	}
};

std::vector<uint8_t> pattern(size_t size, uint8_t seed)
{
	std::vector<uint8_t> value(size);
	for (size_t i = 0; i < size; ++i)
		value[i] = static_cast<uint8_t>(seed + static_cast<uint8_t>(i));
	return value;
}

// First CPU this process is allowed to run on; the cpuset need not include 0.
inline int first_allowed_cpu()
{
	cpu_set_t allowed;
	CPU_ZERO(&allowed);
	REQUIRE(sched_getaffinity(0, sizeof(allowed), &allowed) == 0);
	for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
		if (CPU_ISSET(cpu, &allowed))
			return cpu;
	}
	FAIL("no allowed CPU found");
}
} // namespace

TEST_CASE("LPM trie BPF lookups return SHM-resident per-CPU scratch",
	  "[lpm_trie][inline_value]")
{
	scoped_bpf_cpu guard(first_allowed_cpu());
	for (size_t value_size :
	     { size_t(1), size_t(200), small_value_boundary }) {
		lpm_fixture fixture(value_size);
		auto key = fixture.make_key(10, 0, 0, 1);
		auto value = pattern(value_size, 7);
		REQUIRE(fixture.trie->elem_update(
				reinterpret_cast<bpf_lpm_trie_key *>(key.data()),
				value.data(), 0) == 0);

		void *first = fixture.trie->elem_lookup(
			reinterpret_cast<bpf_lpm_trie_key *>(key.data()));
		REQUIRE(first != nullptr);
		// The BPF-path pointer must live inside SHM so the uBPF runtime
		// bounds check admits BPF dereferences of it.
		REQUIRE(fixture.result_lives_in_shm(first));
		REQUIRE(fixture.result_range_in_shm(first, value_size));
		REQUIRE(std::memcmp(first, value.data(), value_size) == 0);

		// A second lookup returns current bytes after an update,
		// preserving the copy-out snapshot contract.
		auto updated = pattern(value_size, 99);
		REQUIRE(fixture.trie->elem_update(
				reinterpret_cast<bpf_lpm_trie_key *>(key.data()),
				updated.data(), 0) == 0);
		void *second = fixture.trie->elem_lookup(
			reinterpret_cast<bpf_lpm_trie_key *>(key.data()));
		REQUIRE(second != nullptr);
		REQUIRE(fixture.result_lives_in_shm(second));
		REQUIRE(std::memcmp(second, updated.data(), value_size) == 0);
	}
}

TEST_CASE("LPM trie BPF lookups cover values above the small boundary",
	  "[lpm_trie][inline_value]")
{
	scoped_bpf_cpu guard(first_allowed_cpu());
	const size_t value_size = small_value_boundary + 1;
	lpm_fixture fixture(value_size);
	auto key = fixture.make_key(10, 0, 0, 2);
	auto value = pattern(value_size, 31);
	REQUIRE(fixture.trie->elem_update(
			reinterpret_cast<bpf_lpm_trie_key *>(key.data()),
			value.data(), 0) == 0);

	void *result = fixture.trie->elem_lookup(
		reinterpret_cast<bpf_lpm_trie_key *>(key.data()));
	REQUIRE(result != nullptr);
	REQUIRE(fixture.result_lives_in_shm(result));
	REQUIRE(std::memcmp(result, value.data(), value_size) == 0);

	// A map has one fixed value_size, so the 2048-byte case needs its own
	// fixture. It exercises a larger per-CPU scratch slot.
	const size_t large_size = 2048;
	lpm_fixture large_fixture(large_size);
	auto large_key = large_fixture.make_key(10, 0, 0, 3);
	auto large_value = pattern(large_size, 3);
	REQUIRE(large_fixture.trie->elem_update(
			reinterpret_cast<bpf_lpm_trie_key *>(large_key.data()),
			large_value.data(), 0) == 0);
	scoped_bpf_cpu large_guard(first_allowed_cpu());
	void *large_result = large_fixture.trie->elem_lookup(
		reinterpret_cast<bpf_lpm_trie_key *>(large_key.data()));
	REQUIRE(large_result != nullptr);
	REQUIRE(large_fixture.result_lives_in_shm(large_result));
	REQUIRE(std::memcmp(large_result, large_value.data(),
			    large_value.size()) == 0);
}

TEST_CASE("LPM trie BPF scratch slots are isolated per pinned CPU",
	  "[lpm_trie][inline_value]")
{
	// Pick two CPUs this process is actually allowed to run on so the
	// cpuset-derived slot map has entries for both; skip on a single-CPU
	// container where distinct slots cannot exist.
	cpu_set_t allowed;
	CPU_ZERO(&allowed);
	REQUIRE(sched_getaffinity(0, sizeof(allowed), &allowed) == 0);
	std::vector<int> cpus;
	for (int cpu = 0; cpu < CPU_SETSIZE && cpus.size() < 2; ++cpu) {
		if (CPU_ISSET(cpu, &allowed))
			cpus.push_back(cpu);
	}
	if (cpus.size() < 2)
		SKIP();

	const size_t value_size = 64;
	lpm_fixture fixture(value_size);
	auto key = fixture.make_key(172, 16, 0, 1);
	auto value = pattern(value_size, 42);
	REQUIRE(fixture.trie->elem_update(
			reinterpret_cast<bpf_lpm_trie_key *>(key.data()),
			value.data(), 0) == 0);

	void *worker_pointer = nullptr;
	bool worker_value_matches = false;
	std::thread worker([&]() {
		scoped_bpf_cpu worker_guard(cpus[1]);
		worker_pointer = fixture.trie->elem_lookup(
			reinterpret_cast<bpf_lpm_trie_key *>(key.data()));
		worker_value_matches =
			worker_pointer != nullptr &&
			std::memcmp(worker_pointer, value.data(), value_size) == 0;
	});
	void *main_pointer = nullptr;
	{
		scoped_bpf_cpu main_guard(cpus[0]);
		main_pointer = fixture.trie->elem_lookup(
			reinterpret_cast<bpf_lpm_trie_key *>(key.data()));
	}
	worker.join();

	REQUIRE(worker_pointer != nullptr);
	REQUIRE(main_pointer != nullptr);
	// Distinct pinned CPUs map 1:1 to distinct SHM slots.
	REQUIRE(worker_pointer != main_pointer);
	REQUIRE(fixture.result_range_in_shm(worker_pointer, value_size));
	REQUIRE(fixture.result_range_in_shm(main_pointer, value_size));
	REQUIRE(std::memcmp(main_pointer, value.data(), value_size) == 0);
	REQUIRE(worker_value_matches);
}

TEST_CASE("LPM trie same-CPU lookups reuse one copy-out slot",
	  "[lpm_trie][inline_value]")
{
	scoped_bpf_cpu guard(first_allowed_cpu());
	const size_t value_size = 32;
	lpm_fixture fixture(value_size);
	auto key_a = fixture.make_key(10, 0, 0, 1);
	auto key_b = fixture.make_key(10, 0, 0, 2);
	auto value_a = pattern(value_size, 11);
	auto value_b = pattern(value_size, 22);
	auto *ka = reinterpret_cast<bpf_lpm_trie_key *>(key_a.data());
	auto *kb = reinterpret_cast<bpf_lpm_trie_key *>(key_b.data());
	REQUIRE(fixture.trie->elem_update(ka, value_a.data(), 0) == 0);
	REQUIRE(fixture.trie->elem_update(kb, value_b.data(), 0) == 0);

	// One CPU owns one slot: a later lookup on the same CPU reuses it, so the
	// caller must consume a result before issuing another lookup. This
	// documents the single-buffer copy-out contract shared with the old
	// thread-local implementation.
	void *first = fixture.trie->elem_lookup(ka);
	void *second = fixture.trie->elem_lookup(kb);
	REQUIRE(first == second);
	REQUIRE(std::memcmp(second, value_b.data(), value_size) == 0);
}

TEST_CASE("LPM trie syscall lookups use isolated per-thread buffers",
	  "[lpm_trie][inline_value]")
{
	// No pinned BPF CPU: this is the bpf() syscall context, which must keep
	// per-thread buffers so concurrent callers never alias one slot.
	REQUIRE(bpftime_get_current_bpf_cpu() < 0);
	const size_t value_size = 64;
	lpm_fixture fixture(value_size);
	auto key = fixture.make_key(192, 168, 0, 1);
	auto value = pattern(value_size, 5);
	auto *kp = reinterpret_cast<bpf_lpm_trie_key *>(key.data());
	REQUIRE(fixture.trie->elem_update(kp, value.data(), 0) == 0);

	void *main_pointer = fixture.trie->elem_lookup(kp);
	REQUIRE(main_pointer != nullptr);
	REQUIRE(std::memcmp(main_pointer, value.data(), value_size) == 0);

	void *worker_pointer = nullptr;
	bool worker_ok = false;
	std::thread worker([&]() {
		REQUIRE(bpftime_get_current_bpf_cpu() < 0);
		worker_pointer = fixture.trie->elem_lookup(kp);
		worker_ok = worker_pointer != nullptr &&
			    std::memcmp(worker_pointer, value.data(),
					value_size) == 0;
	});
	worker.join();
	REQUIRE(worker_pointer != nullptr);
	REQUIRE(worker_ok);
	// Per-thread TLS buffers must be distinct even if both threads happen to
	// run on the same CPU: the syscall path is not CPU-serialized.
	REQUIRE(worker_pointer != main_pointer);
}
