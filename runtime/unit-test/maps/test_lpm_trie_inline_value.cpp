/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 */
#include "catch2/catch_test_macros.hpp"

#include "bpf_map/userspace/lpm_trie_map.hpp"
#include "../common_def.hpp"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <cstdint>
#include <cstring>
#include <string>
#include <unistd.h>
#include <thread>
#include <vector>

namespace
{
// Must match the inline TLS buffer size in lpm_trie_map_impl::elem_lookup.
constexpr size_t inline_value_capacity = 256;
using bpftime::bpf_lpm_trie_key;

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
		      4 * 1024 * 1024)
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
};

std::vector<uint8_t> pattern(size_t size, uint8_t seed)
{
	std::vector<uint8_t> value(size);
	for (size_t i = 0; i < size; ++i)
		value[i] = static_cast<uint8_t>(seed + static_cast<uint8_t>(i));
	return value;
}
} // namespace

TEST_CASE("LPM trie returns small values through the inline TLS path",
	  "[lpm_trie][inline_value]")
{
	for (size_t value_size :
	     { size_t(1), size_t(200), inline_value_capacity }) {
		lpm_fixture fixture(value_size);
		auto key = fixture.make_key(10, 0, 0, 1);
		auto value = pattern(value_size, 7);
		REQUIRE(fixture.trie->elem_update(
				reinterpret_cast<bpf_lpm_trie_key *>(key.data()),
				value.data(), 0) == 0);

		void *first = fixture.trie->elem_lookup(
			reinterpret_cast<bpf_lpm_trie_key *>(key.data()));
		REQUIRE(first != nullptr);
		REQUIRE(std::memcmp(first, value.data(), value_size) == 0);

		// A second lookup keeps returning the same bytes even when the
		// stored value changes underneath, matching the copy-out
		// contract of the previous vector implementation.
		auto updated = pattern(value_size, 99);
		REQUIRE(fixture.trie->elem_update(
				reinterpret_cast<bpf_lpm_trie_key *>(key.data()),
				updated.data(), 0) == 0);
		void *second = fixture.trie->elem_lookup(
			reinterpret_cast<bpf_lpm_trie_key *>(key.data()));
		REQUIRE(second != nullptr);
		REQUIRE(std::memcmp(second, updated.data(), value_size) == 0);
	}
}

TEST_CASE("LPM trie keeps the vector path for values above the inline limit",
	  "[lpm_trie][inline_value]")
{
	const size_t value_size = inline_value_capacity + 1;
	lpm_fixture fixture(value_size);
	auto key = fixture.make_key(10, 0, 0, 2);
	auto value = pattern(value_size, 31);
	REQUIRE(fixture.trie->elem_update(
			reinterpret_cast<bpf_lpm_trie_key *>(key.data()),
			value.data(), 0) == 0);

	void *result = fixture.trie->elem_lookup(
		reinterpret_cast<bpf_lpm_trie_key *>(key.data()));
	REQUIRE(result != nullptr);
	REQUIRE(std::memcmp(result, value.data(), value_size) == 0);

	// A larger value exercises vector growth beyond one element.
	auto large_key = fixture.make_key(10, 0, 0, 3);
	auto large_value = pattern(2048, 3);
	REQUIRE(fixture.trie->elem_update(
			reinterpret_cast<bpf_lpm_trie_key *>(large_key.data()),
			large_value.data(), 0) == 0);
	void *large_result = fixture.trie->elem_lookup(
		reinterpret_cast<bpf_lpm_trie_key *>(large_key.data()));
	REQUIRE(large_result != nullptr);
	REQUIRE(std::memcmp(large_result, large_value.data(),
			    large_value.size()) == 0);
}

TEST_CASE("LPM trie inline buffers stay independent across threads",
	  "[lpm_trie][inline_value]")
{
	const size_t value_size = 64;
	lpm_fixture fixture(value_size);
	auto key = fixture.make_key(172, 16, 0, 1);
	auto value = pattern(value_size, 42);
	REQUIRE(fixture.trie->elem_update(
			reinterpret_cast<bpf_lpm_trie_key *>(key.data()),
			value.data(), 0) == 0);

	void *worker_pointer = nullptr;
	std::thread worker([&]() {
		worker_pointer = fixture.trie->elem_lookup(
			reinterpret_cast<bpf_lpm_trie_key *>(key.data()));
	});
	void *main_pointer = fixture.trie->elem_lookup(
		reinterpret_cast<bpf_lpm_trie_key *>(key.data()));
	worker.join();

	REQUIRE(worker_pointer != nullptr);
	REQUIRE(main_pointer != nullptr);
	REQUIRE(worker_pointer != main_pointer);
	REQUIRE(std::memcmp(main_pointer, value.data(), value_size) == 0);
	REQUIRE(std::memcmp(worker_pointer, value.data(), value_size) == 0);
}
