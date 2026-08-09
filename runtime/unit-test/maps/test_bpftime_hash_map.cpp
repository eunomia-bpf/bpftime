#include "catch2/catch_message.hpp"
#include "../common_def.hpp"
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <bpf_map/userspace/per_cpu_hash_map.hpp>
#include "bpftime_config.hpp"
#include "bpftime_shm.hpp"
#include "bpftime_handler.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <bpf_map/map_common_def.hpp>
#include "catch2/internal/catch_run_context.hpp"
#include <algorithm>
#include <random>
#include <map>
#include <bpf_map/bpftime_hash_map.hpp>

using namespace boost::interprocess;
using namespace bpftime;

TEST_CASE("bpftime_hash_map basic operations", "[bpftime_hash_map]")
{
	managed_shared_memory segment(open_or_create, "bpftime_hash_map_test",
				      65536);

	size_t num_buckets = 10;
	size_t key_size = 4; // Example key size
	size_t value_size = 8; // Example value size

	bpftime_hash_map map(segment, num_buckets, key_size, value_size);

	SECTION("Insert and Lookup")
	{
		int key1 = 1234;
		int64_t value1 = 5678;
		int key2 = 4321;
		int64_t value2 = 8765;

		REQUIRE(map.elem_update(&key1, &value1) == true);
		REQUIRE(map.elem_update(&key2, &value2) == true);

		REQUIRE(map.elem_lookup(&key1) != nullptr);
		REQUIRE(*(int64_t *)map.elem_lookup(&key1) == value1);

		REQUIRE(map.elem_lookup(&key2) != nullptr);
		REQUIRE(*(int64_t *)map.elem_lookup(&key2) == value2);

		int key3 = 9999;
		REQUIRE(map.elem_lookup(&key3) == nullptr);
	}

	SECTION("Update existing element")
	{
		int key = 1234;
		int64_t value1 = 5678;
		int64_t value2 = 8765;

		REQUIRE(map.elem_update(&key, &value1) == true);
		REQUIRE(map.elem_lookup(&key) != nullptr);
		REQUIRE(*(int64_t *)map.elem_lookup(&key) == value1);

		REQUIRE(map.elem_update(&key, &value2) == true);
		REQUIRE(map.elem_lookup(&key) != nullptr);
		REQUIRE(*(int64_t *)map.elem_lookup(&key) == value2);
	}

	SECTION("Delete element")
	{
		int key1 = 1234;
		int64_t value1 = 5678;
		int key2 = 4321;
		int64_t value2 = 8765;

		REQUIRE(map.elem_update(&key1, &value1) == true);
		REQUIRE(map.elem_update(&key2, &value2) == true);

		REQUIRE(map.elem_delete(&key1) == true);
		REQUIRE(map.elem_lookup(&key1) == nullptr);
		REQUIRE(map.elem_lookup(&key2) != nullptr);
		REQUIRE(*(int64_t *)map.elem_lookup(&key2) == value2);

		REQUIRE(map.elem_delete(&key2) == true);
		REQUIRE(map.elem_lookup(&key2) == nullptr);
	}

	SECTION("Get element count")
	{
		int key1 = 1234;
		int64_t value1 = 5678;
		int key2 = 4321;
		int64_t value2 = 8765;

		REQUIRE(map.get_elem_count() == 0);

		REQUIRE(map.elem_update(&key1, &value1) == true);
		REQUIRE(map.get_elem_count() == 1);

		REQUIRE(map.elem_update(&key2, &value2) == true);
		REQUIRE(map.get_elem_count() == 2);

		REQUIRE(map.elem_delete(&key1) == true);
		REQUIRE(map.get_elem_count() == 1);

		REQUIRE(map.elem_delete(&key2) == true);
		REQUIRE(map.get_elem_count() == 0);
	}

	SECTION("Insert more elements than num_buckets")
	{
		int key;
		int64_t value;

		for (size_t i = 0; i < num_buckets; ++i) {
			key = i;
			value = i * 100;
			REQUIRE(map.elem_update(&key, &value) == true);
		}

		key = num_buckets;
		value = num_buckets * 100;
		REQUIRE(map.elem_update(&key, &value) == false); // Should fail
								 // as the map
								 // is full

		for (size_t i = 0; i < num_buckets; ++i) {
			key = i;
			REQUIRE(map.elem_lookup(&key) != nullptr);
			REQUIRE(*(int64_t *)map.elem_lookup(&key) ==
				(int64_t)i * 100);
		}
	}

	SECTION("Insert, delete, and re-insert elements")
	{
		int key1 = 1234;
		int64_t value1 = 5678;
		int key2 = 4321;
		int64_t value2 = 8765;
		int key3 = 5678;
		int64_t value3 = 4321;

		REQUIRE(map.elem_update(&key1, &value1) == true);
		REQUIRE(map.elem_update(&key2, &value2) == true);

		REQUIRE(map.elem_delete(&key1) == true);
		REQUIRE(map.elem_lookup(&key1) == nullptr);

		REQUIRE(map.elem_update(&key3, &value3) == true);
		REQUIRE(map.elem_lookup(&key3) != nullptr);
		REQUIRE(*(int64_t *)map.elem_lookup(&key3) == value3);

		REQUIRE(map.elem_lookup(&key2) != nullptr);
		REQUIRE(*(int64_t *)map.elem_lookup(&key2) == value2);
	}
}
