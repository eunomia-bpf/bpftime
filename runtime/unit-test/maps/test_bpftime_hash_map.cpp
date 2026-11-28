#include "catch2/catch_message.hpp"
#include "../common_def.hpp"
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <bpf_map/userspace/per_cpu_hash_map.hpp>
#include "bpftime_config.hpp"
#include "bpftime_shm.hpp"
#include "bpftime_handler.hpp"
#include <catch2/catch_test_macros.hpp>
#include <csignal>
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

	SECTION("Delete preserves a colliding probe chain")
	{
		// With the 11-bucket table used by this fixture, these integer keys
		// hash to the same initial bucket on little-endian platforms.
		int key1 = 1;
		int key2 = 12;
		int key3 = 23;
		int64_t value1 = 101;
		int64_t value2 = 112;
		int64_t value3 = 123;

		REQUIRE(map.elem_update(&key1, &value1) == true);
		REQUIRE(map.elem_update(&key2, &value2) == true);
		REQUIRE(map.elem_delete(&key1) == true);

		auto *second = static_cast<int64_t *>(map.elem_lookup(&key2));
		REQUIRE(second != nullptr);
		REQUIRE(*second == value2);

		// A later insertion may reuse the tombstone without making the
		// existing colliding key unreachable.
		REQUIRE(map.elem_update(&key3, &value3) == true);
		second = static_cast<int64_t *>(map.elem_lookup(&key2));
		auto *third = static_cast<int64_t *>(map.elem_lookup(&key3));
		REQUIRE(second != nullptr);
		REQUIRE(*second == value2);
		REQUIRE(third != nullptr);
		REQUIRE(*third == value3);
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

	SECTION("Tombstone-free O(1) reuse under high delete churn")
	{
		for (int i = 0; i < 5000; ++i) {
			int64_t v = i * 10;
			REQUIRE(map.elem_update(&i, &v));
			REQUIRE(map.elem_delete(&i));
		}
		int absent = 999999;
		REQUIRE((map.get_elem_count() == 0 && !map.elem_lookup(&absent)));
		for (int i = 0; i < (int)num_buckets; ++i)
			REQUIRE(map.elem_update(&i, &absent));
	}
	shared_memory_object::remove("bpftime_hash_map_test");
}

#if defined(__linux__)
#include <sys/wait.h>

TEST_CASE("bpftime_hash_map recovers stripe after child dies mid-upsert or mid-erase",
	  "[bpftime_hash_map][dead_owner]")
{
	const std::string shm_name =
		"bpftime_hmap_recover_" + std::to_string(getpid());
	::shm_remove remover(shm_name.c_str());
	managed_shared_memory segment(create_only, shm_name.c_str(), 1 << 20);
	const long page_sz = sysconf(_SC_PAGESIZE);
	REQUIRE(page_sz >= 4096);
	const size_t val_sz = static_cast<size_t>(page_sz) * 2;
	auto *map = segment.construct<bpftime_hash_map>("hmap")(
		segment, 4, sizeof(uint32_t), val_sz);
	REQUIRE(map != nullptr);

	auto make_half_guarded_buf = [&]() -> uint8_t * {
		void *p = mmap(nullptr, val_sz, PROT_READ | PROT_WRITE,
			       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		REQUIRE(p != MAP_FAILED);
		REQUIRE(mprotect(static_cast<uint8_t *>(p) + page_sz, page_sz,
				 PROT_NONE) == 0);
		return static_cast<uint8_t *>(p);
	};

	// 1. Child dies with SIGSEGV inside upsert() while holding the stripe lock,
	//    after counting and claiming a slot but before publishing it. The
	//    non-blocking path recovers the stripe; the claimed slot stays leaked.
	pid_t c1 = fork();
	REQUIRE(c1 >= 0);
	if (c1 == 0) {
		std::signal(SIGSEGV, SIG_DFL);
		uint8_t *bad_in = make_half_guarded_buf();
		uint32_t k = 7;
		(void)map->upsert(&k, bad_in, 0, true, 0, val_sz);
		_exit(0);
	}
	int st1 = 0;
	REQUIRE(waitpid(c1, &st1, 0) == c1);
	REQUIRE(WIFSIGNALED(st1));

	std::vector<uint8_t> good_val(val_sz, 0x5a);
	uint32_t k7 = 7;
	REQUIRE(map->elem_lookup(&k7) == nullptr);
	REQUIRE(map->upsert(&k7, good_val.data(), 0, false, 0, val_sz) == 0);
	REQUIRE(map->get_elem_count() == 2);

	// 2. Child dies with SIGSEGV inside erase() while holding the stripe lock,
	//    while copying the value out, before unlinking. The element is intact;
	//    the blocking syscall path recovers the stripe and deletes it.
	pid_t c2 = fork();
	REQUIRE(c2 >= 0);
	if (c2 == 0) {
		std::signal(SIGSEGV, SIG_DFL);
		uint8_t *bad_out = make_half_guarded_buf();
		uint32_t k = 7;
		(void)map->erase(&k, bad_out, true);
		_exit(0);
	}
	int st2 = 0;
	REQUIRE(waitpid(c2, &st2, 0) == c2);
	REQUIRE(WIFSIGNALED(st2));

	REQUIRE(map->elem_lookup(&k7) != nullptr);
	REQUIRE(map->erase(&k7, nullptr, true) == 0);
	REQUIRE(map->elem_lookup(&k7) == nullptr);
	REQUIRE(map->get_elem_count() == 1);

	// Both stripes keep working. The one slot leaked by the dead inserter
	// leaves this 4-slot map room for 3 elements.
	for (uint32_t k = 0; k < 3; ++k)
		REQUIRE(map->upsert(&k, good_val.data(), 0, false, 0, val_sz) ==
			0);
	uint32_t k3 = 3;
	REQUIRE(map->upsert(&k3, good_val.data(), 0, false, 0, val_sz) == -1);
	REQUIRE(errno == E2BIG);
	REQUIRE(map->get_elem_count() == 4);
}
#endif