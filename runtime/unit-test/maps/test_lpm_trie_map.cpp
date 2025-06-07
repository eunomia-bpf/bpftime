#include "catch2/catch_test_macros.hpp"
#include "catch2/catch_message.hpp"

#include "bpf_map/userspace/lpm_trie_map.hpp"
#include <boost/interprocess/managed_shared_memory.hpp>
#include <cstring>
#include <cerrno>
#include <arpa/inet.h>

// Helper structure for IPv4 testing
struct ipv4_lpm_key {
	uint32_t prefixlen;
	uint32_t data; // IPv4 address in network byte order
};

TEST_CASE("LPM Trie Map Constructor Validation", "[lpm_trie_map][constructor]")
{
	const char *SHARED_MEMORY_NAME = "LPMTrieConstructorTestShm";
	const size_t SHARED_MEMORY_SIZE = 1024;

	struct ShmRemover {
		const char *name;
		ShmRemover(const char *n) : name(n)
		{
			boost::interprocess::shared_memory_object::remove(name);
		}
		~ShmRemover()
		{
			boost::interprocess::shared_memory_object::remove(name);
		}
	} remover(SHARED_MEMORY_NAME);

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME,
		SHARED_MEMORY_SIZE);

	// Test invalid key_size (too small)
	REQUIRE_THROWS_AS(bpftime::lpm_trie_map_impl(shm, 3, 4, 10),
			  std::invalid_argument);

	// Test invalid value_size (0)
	REQUIRE_THROWS_AS(bpftime::lpm_trie_map_impl(shm, 8, 0, 10),
			  std::invalid_argument);

	// Test invalid max_entries (0)
	REQUIRE_THROWS_AS(bpftime::lpm_trie_map_impl(shm, 8, 4, 0),
			  std::invalid_argument);

	// Test valid construction
	REQUIRE_NOTHROW(bpftime::lpm_trie_map_impl(shm, 8, 4, 10));
}

TEST_CASE("LPM Trie Map IPv4 Operations", "[lpm_trie_map][ipv4]")
{
	const char *SHARED_MEMORY_NAME = "LPMTrieIPv4TestShm";
	const size_t SHARED_MEMORY_SIZE = 65536;

	struct ShmRemover {
		const char *name;
		ShmRemover(const char *n) : name(n)
		{
			boost::interprocess::shared_memory_object::remove(name);
		}
		~ShmRemover()
		{
			boost::interprocess::shared_memory_object::remove(name);
		}
	} remover(SHARED_MEMORY_NAME);

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME,
		SHARED_MEMORY_SIZE);

	// Create LPM Trie for IPv4 (key_size = 4 bytes prefixlen + 4 bytes
	// IPv4)
	bpftime::lpm_trie_map_impl *trie = nullptr;
	try {
		trie = shm.construct<bpftime::lpm_trie_map_impl>(
			"LPMTrieInstance")(shm, sizeof(ipv4_lpm_key),
					   sizeof(uint32_t), 10);
		REQUIRE(trie != nullptr);
	} catch (const std::exception &ex) {
		FAIL("Failed to construct LPM Trie: " << ex.what());
	}

	SECTION("Basic IPv4 Prefix Operations")
	{
		// Test data: 192.168.0.0/16 -> value 1
		ipv4_lpm_key key1;
		key1.prefixlen = 16;
		key1.data = htonl(0xC0A80000); // 192.168.0.0 in network byte
					       // order
		uint32_t value1 = 1;

		// Insert first prefix
		REQUIRE(trie->elem_update(&key1, &value1, 0) == 0);
		REQUIRE(trie->get_current_entries() == 1);

		// Lookup exact match
		ipv4_lpm_key lookup_key;
		lookup_key.prefixlen = 32; // Full address lookup
		lookup_key.data = htonl(0xC0A80001); // 192.168.0.1

		void *result = trie->elem_lookup(&lookup_key);
		REQUIRE(result != nullptr);
		REQUIRE(*(uint32_t *)result == value1);

		// Test data: 192.168.1.0/24 -> value 2 (more specific)
		ipv4_lpm_key key2;
		key2.prefixlen = 24;
		key2.data = htonl(0xC0A80100); // 192.168.1.0
		uint32_t value2 = 2;

		REQUIRE(trie->elem_update(&key2, &value2, 0) == 0);
		REQUIRE(trie->get_current_entries() == 2);

		// Lookup should return more specific match
		lookup_key.data = htonl(0xC0A80101); // 192.168.1.1
		result = trie->elem_lookup(&lookup_key);
		REQUIRE(result != nullptr);
		REQUIRE(*(uint32_t *)result == value2); // Should match /24
							// prefix

		// Lookup in /16 but not /24 range
		lookup_key.data = htonl(0xC0A80201); // 192.168.2.1
		result = trie->elem_lookup(&lookup_key);
		REQUIRE(result != nullptr);
		REQUIRE(*(uint32_t *)result == value1); // Should match /16
							// prefix
	}

	SECTION("Prefix Deletion")
	{
		// Insert test data
		ipv4_lpm_key key;
		key.prefixlen = 24;
		key.data = htonl(0xC0A80000); // 192.168.0.0/24
		uint32_t value = 100;

		REQUIRE(trie->elem_update(&key, &value, 0) == 0);

		// Verify insertion
		void *result = trie->elem_lookup(&key);
		REQUIRE(result != nullptr);
		REQUIRE(*(uint32_t *)result == value);

		// Delete the prefix
		REQUIRE(trie->elem_delete(&key) == 0);
		REQUIRE(trie->get_current_entries() == 0);

		// Verify deletion
		errno = 0;
		result = trie->elem_lookup(&key);
		REQUIRE(result == nullptr);
		REQUIRE(errno == ENOENT);
	}

	SECTION("Invalid Operations")
	{
		ipv4_lpm_key key;
		key.prefixlen = 40; // Invalid prefix length (> 32 for IPv4)
		key.data = htonl(0xC0A80000);
		uint32_t value = 1;

		// Invalid prefix length
		errno = 0;
		REQUIRE(trie->elem_update(&key, &value, 0) == -1);
		REQUIRE(errno == EINVAL);

		// Null key
		errno = 0;
		REQUIRE(trie->elem_lookup(nullptr) == nullptr);
		REQUIRE(errno == EINVAL);

		// Null value
		errno = 0;
		key.prefixlen = 16;
		REQUIRE(trie->elem_update(&key, nullptr, 0) == -1);
		REQUIRE(errno == EINVAL);
	}

	SECTION("Unsupported Operations")
	{
		uint32_t value = 1;

		// Push/Pop/Peek operations should not be supported
		errno = 0;
		REQUIRE(trie->map_push_elem(&value, 0) == -1);
		REQUIRE(errno == ENOTSUP);

		errno = 0;
		REQUIRE(trie->map_pop_elem(&value) == -1);
		REQUIRE(errno == ENOTSUP);

		errno = 0;
		REQUIRE(trie->map_peek_elem(&value) == -1);
		REQUIRE(errno == ENOTSUP);
	}

	// Cleanup
	if (trie) {
		shm.destroy_ptr(trie);
	}
}

TEST_CASE("LPM Trie Map Longest Prefix Match", "[lpm_trie_map][lpm]")
{
	const char *SHARED_MEMORY_NAME = "LPMTrieLPMTestShm";
	const size_t SHARED_MEMORY_SIZE = 65536;

	struct ShmRemover {
		const char *name;
		ShmRemover(const char *n) : name(n)
		{
			boost::interprocess::shared_memory_object::remove(name);
		}
		~ShmRemover()
		{
			boost::interprocess::shared_memory_object::remove(name);
		}
	} remover(SHARED_MEMORY_NAME);

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME,
		SHARED_MEMORY_SIZE);

	bpftime::lpm_trie_map_impl *trie = nullptr;
	try {
		trie = shm.construct<bpftime::lpm_trie_map_impl>(
			"LPMTrieLPMInstance")(shm, sizeof(ipv4_lpm_key),
					      sizeof(uint32_t), 20);
		REQUIRE(trie != nullptr);
	} catch (const std::exception &ex) {
		FAIL("Failed to construct LPM Trie: " << ex.what());
	}

	// Build a trie with multiple overlapping prefixes
	// 10.0.0.0/8 -> value 1
	// 10.1.0.0/16 -> value 2
	// 10.1.1.0/24 -> value 3
	// 10.1.1.128/25 -> value 4

	ipv4_lpm_key keys[4];
	uint32_t values[4] = { 1, 2, 3, 4 };

	// 10.0.0.0/8
	keys[0].prefixlen = 8;
	keys[0].data = htonl(0x0A000000);

	// 10.1.0.0/16
	keys[1].prefixlen = 16;
	keys[1].data = htonl(0x0A010000);

	// 10.1.1.0/24
	keys[2].prefixlen = 24;
	keys[2].data = htonl(0x0A010100);

	// 10.1.1.128/25
	keys[3].prefixlen = 25;
	keys[3].data = htonl(0x0A010180);

	// Insert all prefixes
	for (int i = 0; i < 4; i++) {
		REQUIRE(trie->elem_update(&keys[i], &values[i], 0) == 0);
	}

	REQUIRE(trie->get_current_entries() == 4);

	// Test longest prefix matching
	ipv4_lpm_key lookup_key;
	lookup_key.prefixlen = 32; // Full address lookup

	// Test 10.1.1.200 -> should match 10.1.1.128/25 (value 4)
	lookup_key.data = htonl(0x0A0101C8); // 10.1.1.200
	void *result = trie->elem_lookup(&lookup_key);
	REQUIRE(result != nullptr);
	REQUIRE(*(uint32_t *)result == 4);

	// Test 10.1.1.50 -> should match 10.1.1.0/24 (value 3)
	lookup_key.data = htonl(0x0A010132); // 10.1.1.50
	result = trie->elem_lookup(&lookup_key);
	REQUIRE(result != nullptr);
	REQUIRE(*(uint32_t *)result == 3);

	// Test 10.1.2.1 -> should match 10.1.0.0/16 (value 2)
	lookup_key.data = htonl(0x0A010201); // 10.1.2.1
	result = trie->elem_lookup(&lookup_key);
	REQUIRE(result != nullptr);
	REQUIRE(*(uint32_t *)result == 2);

	// Test 10.2.0.1 -> should match 10.0.0.0/8 (value 1)
	lookup_key.data = htonl(0x0A020001); // 10.2.0.1
	result = trie->elem_lookup(&lookup_key);
	REQUIRE(result != nullptr);
	REQUIRE(*(uint32_t *)result == 1);

	// Test 192.168.1.1 -> should not match anything
	lookup_key.data = htonl(0xC0A80101); // 192.168.1.1
	errno = 0;
	result = trie->elem_lookup(&lookup_key);
	REQUIRE(result == nullptr);
	REQUIRE(errno == ENOENT);

	// Cleanup
	if (trie) {
		shm.destroy_ptr(trie);
	}
}