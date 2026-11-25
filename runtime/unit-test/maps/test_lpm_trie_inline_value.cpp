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

TEST_CASE("LPM trie lookups of different keys stay distinct",
	  "[lpm_trie][value_lifetime]")
{
	for (size_t value_size : { size_t(1), size_t(256), size_t(2048) }) {
		lpm_fixture fixture(value_size);
		auto key1 = fixture.make_key(10, 0, 0, 1);
		auto key2 = fixture.make_key(10, 0, 0, 2);
		auto value1 = pattern(value_size, 7);
		auto value2 = pattern(value_size, 99);
		REQUIRE(fixture.trie->elem_update(
				reinterpret_cast<bpf_lpm_trie_key *>(key1.data()),
				value1.data(), 0) == 0);
		REQUIRE(fixture.trie->elem_update(
				reinterpret_cast<bpf_lpm_trie_key *>(key2.data()),
				value2.data(), 0) == 0);

		void *first = fixture.trie->elem_lookup(
			reinterpret_cast<bpf_lpm_trie_key *>(key1.data()));
		void *second = fixture.trie->elem_lookup(
			reinterpret_cast<bpf_lpm_trie_key *>(key2.data()));
		REQUIRE(first != nullptr);
		REQUIRE(second != nullptr);

		// The kernel keeps both results valid: a second lookup must not
		// clobber the first one.
		REQUIRE(first != second);
		REQUIRE(std::memcmp(first, value1.data(), value_size) == 0);
		REQUIRE(std::memcmp(second, value2.data(), value_size) == 0);
	}
}

TEST_CASE("LPM trie repeats a lookup at the same address",
	  "[lpm_trie][value_lifetime]")
{
	for (size_t value_size : { size_t(1), size_t(256), size_t(2048) }) {
		lpm_fixture fixture(value_size);
		auto key = fixture.make_key(10, 0, 0, 1);
		auto value = pattern(value_size, 31);
		REQUIRE(fixture.trie->elem_update(
				reinterpret_cast<bpf_lpm_trie_key *>(key.data()),
				value.data(), 0) == 0);

		void *first = fixture.trie->elem_lookup(
			reinterpret_cast<bpf_lpm_trie_key *>(key.data()));
		void *second = fixture.trie->elem_lookup(
			reinterpret_cast<bpf_lpm_trie_key *>(key.data()));
		REQUIRE(first != nullptr);
		REQUIRE(second != nullptr);
		REQUIRE(first == second);
		REQUIRE(std::memcmp(second, value.data(), value_size) == 0);

		// A hold on the result survives an update of the same key, as it
		// does in the kernel.
		void *held = first;
		auto updated = pattern(value_size, 55);
		REQUIRE(fixture.trie->elem_update(
				reinterpret_cast<bpf_lpm_trie_key *>(key.data()),
				updated.data(), 0) == 0);
		REQUIRE(std::memcmp(held, updated.data(), value_size) == 0);
	}
}

TEST_CASE("LPM trie value pointers agree across threads",
	  "[lpm_trie][value_lifetime]")
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

	// One value has one address, whichever thread looks it up, and it
	// outlives the thread that did.
	REQUIRE(worker_pointer != nullptr);
	REQUIRE(main_pointer != nullptr);
	REQUIRE(worker_pointer == main_pointer);
	REQUIRE(std::memcmp(main_pointer, value.data(), value_size) == 0);
}
