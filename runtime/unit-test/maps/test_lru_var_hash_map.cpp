#include "catch2/catch_test_macros.hpp"
#include "linux/bpf.h"
#include "unit-test/common_def.hpp"
#include <bpf_map/userspace/lru_var_hash_map.hpp>
#include <memory>
static const char *SHM_NAME = "_HASH_MAP_TEST";

using namespace boost::interprocess;
using namespace bpftime;

struct shm_initializer {
	std::unique_ptr<shm_remove> remover;
	std::unique_ptr<managed_shared_memory> mem;
	shm_initializer(const shm_initializer &) = delete;
	shm_initializer &operator=(const shm_initializer &) = delete;
	shm_initializer(const char *memory_name, bool should_create,
			size_t memory_size = 20 << 20)
	{
		if (should_create) {
			remover = std::make_unique<shm_remove>(memory_name);
			mem = std::make_unique<managed_shared_memory>(
				boost::interprocess::create_only, memory_name,
				memory_size);
		} else {
			mem = std::make_unique<managed_shared_memory>(
				boost::interprocess::open_only, memory_name);
		}
	}
};

TEST_CASE("Test basic lru map operations")
{
	shm_initializer shm(SHM_NAME, true);
	auto &mem = *shm.mem;
	{
		lru_var_hash_map_impl map(mem, 4, 4, 1000);
		for (int i = 0; i < 500; i++) {
			int key = i;
			int value = 500 + i;
			REQUIRE(map.elem_update(&key, &value, BPF_NOEXIST) ==
				0);
		}
		for (int i = 500 - 1; i >= 0; i--) {
			auto ptr = (int *)map.elem_lookup(&i);
			REQUIRE(ptr != nullptr);
			REQUIRE(*ptr == i + 500);
		}

		for (int i = 0; i < 500; i++) {
			int key = i;
			int value = 5000 + i;
			REQUIRE(map.elem_update(&key, &value, BPF_EXIST) == 0);
		}
		for (int i = 500 - 1; i >= 0; i--) {
			auto ptr = (int *)map.elem_lookup(&i);
			REQUIRE(ptr != nullptr);
			REQUIRE(*ptr == i + 5000);
		}
		// Remove some elements
		for (int i = 0; i < 500; i++) {
			if (i % 2 == 0) {
				REQUIRE(map.elem_delete(&i) == 0);
			}
		}
		for (int i = 500 - 1; i >= 0; i--) {
			auto ptr = (int *)map.elem_lookup(&i);
			if (i % 2 == 0) {
				REQUIRE(ptr == nullptr);
			} else {
				REQUIRE(ptr != nullptr);
				REQUIRE(*ptr == i + 5000);
			}
		}
	}
	{
		lru_var_hash_map_impl map(mem, 4, 4, 5);
		// Test LRU operations
		for (int i = 0; i < 5; i++) {
			REQUIRE(map.elem_update(&i, &i, BPF_NOEXIST) == 0);
		}
		// Access elements except key=3
		for (int i = 0; i < 5; i++) {
			if (i != 3) {
				auto ptr = (int *)map.elem_lookup(&i);
				REQUIRE(ptr != nullptr);
				REQUIRE(*ptr == i);
			}
		}
		// Insert a new element
		int key = 10;
		REQUIRE(map.elem_update(&key, &key, BPF_NOEXIST) == 0);
		{
			auto ptr = (int *)map.elem_lookup(&key);
			REQUIRE(ptr != nullptr);
			REQUIRE(*ptr == key);
		}
		key = 3;
		REQUIRE(map.elem_lookup(&key) == nullptr);
	}
}
