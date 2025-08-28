/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, bpftime project
 * All rights reserved.
 */

#include "catch2/catch_test_macros.hpp"
#include "catch2/catch_message.hpp"

#include "bpf_map/userspace/lru_var_hash_map.hpp"
#include "../common_def.hpp"
#include <boost/interprocess/managed_shared_memory.hpp>
#include <cstring>
#include <cerrno>
#include <cstdlib>
#include <set>
#include "linux/bpf.h"

// Test structure for LRU hash map testing (kept minimal)
struct lru_test_key {
	uint32_t key;
};

struct lru_test_value {
	uint64_t value;
};

// Basic LRU Hash Map functionality tests
TEST_CASE("LRU Hash Map Basic Operations", "[lru_hash][basic]")
{
	const char *SHARED_MEMORY_NAME = "LRUHashMapBasicTestShm";
	const size_t SHARED_MEMORY_SIZE = 1024 * 1024;

	shm_remove remover((std::string(SHARED_MEMORY_NAME)));

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME,
		SHARED_MEMORY_SIZE);

	// Create LRU hash map with small capacity to test eviction
	const size_t map_capacity = 10;
	bpftime::lru_var_hash_map_impl *lru_map = nullptr;

	try {
		lru_map = shm.construct<bpftime::lru_var_hash_map_impl>(
			"LRUHashMapBasicInstance")(
			shm, sizeof(uint32_t), sizeof(uint64_t), map_capacity);
		REQUIRE(lru_map != nullptr);
	} catch (const std::exception &ex) {
		FAIL("Failed to construct LRU Hash Map: " << ex.what());
	}

	SECTION("Insert and Lookup Operations")
	{
		// 插入一些元素
		for (uint32_t i = 0; i < map_capacity; i++) {
			uint64_t value = i * 100;
			REQUIRE(lru_map->elem_update(&i, &value, BPF_NOEXIST) ==
				0);
		}

		// 验证所有元素都能找到
		for (uint32_t i = 0; i < map_capacity; i++) {
			void *result = lru_map->elem_lookup(&i);
			REQUIRE(result != nullptr);
			REQUIRE(*(uint64_t *)result == i * 100);
		}
	}

	SECTION("LRU Eviction Test")
	{
		// 填充到容量上限
		for (uint32_t i = 0; i < map_capacity; i++) {
			uint64_t value = i;
			REQUIRE(lru_map->elem_update(&i, &value, BPF_NOEXIST) ==
				0);
		}

		// 访问除了key=0之外的所有元素，使key=0成为LRU
		for (uint32_t i = 1; i < map_capacity; i++) {
			void *result = lru_map->elem_lookup(&i);
			REQUIRE(result != nullptr);
		}

		// 插入新元素，应该evict key=0
		uint32_t new_key = map_capacity;
		uint64_t new_value = 999;
		REQUIRE(lru_map->elem_update(&new_key, &new_value,
					     BPF_NOEXIST) == 0);

		// 验证新元素存在
		void *result = lru_map->elem_lookup(&new_key);
		REQUIRE(result != nullptr);
		REQUIRE(*(uint64_t *)result == new_value);

		// 验证key=0被evicted
		uint32_t evicted_key = 0;
		errno = 0;
		result = lru_map->elem_lookup(&evicted_key);
		REQUIRE(result == nullptr);
		REQUIRE(errno == ENOENT);
	}

	SECTION("Update and Delete Operations")
	{
		uint32_t key = 42;
		uint64_t value1 = 123;
		uint64_t value2 = 456;

		// 插入元素
		REQUIRE(lru_map->elem_update(&key, &value1, BPF_NOEXIST) == 0);

		// 更新元素
		REQUIRE(lru_map->elem_update(&key, &value2, BPF_EXIST) == 0);

		void *result = lru_map->elem_lookup(&key);
		REQUIRE(result != nullptr);
		REQUIRE(*(uint64_t *)result == value2);

		// 删除元素
		REQUIRE(lru_map->elem_delete(&key) == 0);

		errno = 0;
		result = lru_map->elem_lookup(&key);
		REQUIRE(result == nullptr);
		REQUIRE(errno == ENOENT);

		// 重复删除应该失败
		errno = 0;
		REQUIRE(lru_map->elem_delete(&key) == -1);
		REQUIRE(errno == ENOENT);
	}

	// Cleanup
	if (lru_map) {
		shm.destroy_ptr(lru_map);
	}
}

// Update flags test (BPF_EXIST, BPF_NOEXIST, BPF_ANY)
TEST_CASE("LRU Hash Map Update Flags", "[lru_hash][flags]")
{
	const char *SHARED_MEMORY_NAME = "LRUHashMapFlagsTestShm";
	const size_t SHARED_MEMORY_SIZE = 1024 * 1024;

	shm_remove remover((std::string(SHARED_MEMORY_NAME)));

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME,
		SHARED_MEMORY_SIZE);

	bpftime::lru_var_hash_map_impl *lru_map = nullptr;

	try {
		lru_map = shm.construct<bpftime::lru_var_hash_map_impl>(
			"LRUHashMapFlagsInstance")(shm, sizeof(uint32_t),
						   sizeof(uint64_t), 10);
		REQUIRE(lru_map != nullptr);
	} catch (const std::exception &ex) {
		FAIL("Failed to construct LRU Hash Map: " << ex.what());
	}

	uint32_t key = 123;
	uint64_t value1 = 456;
	uint64_t value2 = 789;

	SECTION("BPF_NOEXIST flag tests")
	{
		// 在空map中用BPF_NOEXIST插入应该成功
		REQUIRE(lru_map->elem_update(&key, &value1, BPF_NOEXIST) == 0);

		void *result = lru_map->elem_lookup(&key);
		REQUIRE(result != nullptr);
		REQUIRE(*(uint64_t *)result == value1);

		// 重复插入相同key应该失败
		errno = 0;
		REQUIRE(lru_map->elem_update(&key, &value2, BPF_NOEXIST) == -1);
		REQUIRE(errno == EEXIST);
	}

	SECTION("BPF_EXIST flag tests")
	{
		// 先插入一个元素
		REQUIRE(lru_map->elem_update(&key, &value1, BPF_NOEXIST) == 0);

		// 用BPF_EXIST更新应该成功
		REQUIRE(lru_map->elem_update(&key, &value2, BPF_EXIST) == 0);

		void *result = lru_map->elem_lookup(&key);
		REQUIRE(result != nullptr);
		REQUIRE(*(uint64_t *)result == value2);

		// 删除元素
		REQUIRE(lru_map->elem_delete(&key) == 0);

		// 对不存在的key用BPF_EXIST应该失败
		errno = 0;
		REQUIRE(lru_map->elem_update(&key, &value1, BPF_EXIST) == -1);
		REQUIRE(errno == ENOENT);
	}

	SECTION("BPF_ANY flag tests")
	{
		// BPF_ANY对不存在的key应该成功
		REQUIRE(lru_map->elem_update(&key, &value1, BPF_ANY) == 0);

		// BPF_ANY对已存在的key也应该成功
		REQUIRE(lru_map->elem_update(&key, &value2, BPF_ANY) == 0);

		void *result = lru_map->elem_lookup(&key);
		REQUIRE(result != nullptr);
		REQUIRE(*(uint64_t *)result == value2);
	}

	// Cleanup
	if (lru_map) {
		shm.destroy_ptr(lru_map);
	}
}

// Capacity and iteration tests
TEST_CASE("LRU Hash Map Capacity Limits", "[lru_hash][capacity]")
{
	const char *SHARED_MEMORY_NAME = "LRUHashMapCapacityTestShm";
	const size_t SHARED_MEMORY_SIZE = 1024 * 1024;

	shm_remove remover((std::string(SHARED_MEMORY_NAME)));

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME,
		SHARED_MEMORY_SIZE);

	const size_t small_capacity = 5;
	bpftime::lru_var_hash_map_impl *lru_map = nullptr;

	try {
		lru_map = shm.construct<bpftime::lru_var_hash_map_impl>(
			"LRUHashMapCapacityInstance")(shm, sizeof(uint32_t),
						      sizeof(uint64_t),
						      small_capacity);
		REQUIRE(lru_map != nullptr);
	} catch (const std::exception &ex) {
		FAIL("Failed to construct LRU Hash Map: " << ex.what());
	}

	// 填充到容量上限
	for (uint32_t i = 0; i < small_capacity; i++) {
		uint64_t value = i * 10;
		REQUIRE(lru_map->elem_update(&i, &value, BPF_NOEXIST) == 0);
	}

	// 验证所有元素存在
	for (uint32_t i = 0; i < small_capacity; i++) {
		void *result = lru_map->elem_lookup(&i);
		REQUIRE(result != nullptr);
		REQUIRE(*(uint64_t *)result == i * 10);
	}

	// 插入额外元素应该触发LRU eviction，而不是失败
	uint32_t extra_key = small_capacity;
	uint64_t extra_value = 999;
	REQUIRE(lru_map->elem_update(&extra_key, &extra_value, BPF_NOEXIST) ==
		0);

	// 验证新元素存在
	void *result = lru_map->elem_lookup(&extra_key);
	REQUIRE(result != nullptr);
	REQUIRE(*(uint64_t *)result == extra_value);

	// Cleanup
	if (lru_map) {
		shm.destroy_ptr(lru_map);
	}
}

TEST_CASE("LRU Hash Map Iteration", "[lru_hash][iterator]")
{
	const char *SHARED_MEMORY_NAME = "LRUHashMapIterTestShm";
	const size_t SHARED_MEMORY_SIZE = 1024 * 1024;

	shm_remove remover((std::string(SHARED_MEMORY_NAME)));

	boost::interprocess::managed_shared_memory shm(
		boost::interprocess::create_only, SHARED_MEMORY_NAME,
		SHARED_MEMORY_SIZE);

	const size_t map_capacity = 8;
	bpftime::lru_var_hash_map_impl *lru_map = nullptr;

	try {
		lru_map = shm.construct<bpftime::lru_var_hash_map_impl>(
			"LRUHashMapIterInstance")(
			shm, sizeof(uint32_t), sizeof(uint64_t), map_capacity);
		REQUIRE(lru_map != nullptr);
	} catch (const std::exception &ex) {
		FAIL("Failed to construct LRU Hash Map: " << ex.what());
	}

	SECTION("Iterate all keys with map_get_next_key")
	{
		// empty map: first next should fail
		uint32_t next_key = 0;
		errno = 0;
		REQUIRE(lru_map->map_get_next_key(nullptr, &next_key) == -1);
		REQUIRE(errno == ENOENT);

		// fill some keys
		uint32_t keys[] = { 10, 20, 30, 40 };
		for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
			uint64_t value = keys[i] * 2;
			REQUIRE(lru_map->elem_update(&keys[i], &value,
						     BPF_NOEXIST) == 0);
		}

		// start iteration from nullptr
		uint32_t cur = 0;
		std::set<uint32_t> visited;
		int ret = lru_map->map_get_next_key(nullptr, &cur);
		while (ret == 0) {
			visited.insert(cur);
			ret = lru_map->map_get_next_key(&cur, &cur);
		}
		REQUIRE(errno == ENOENT);

		// verify all inserted keys visited (order unspecified)
		for (auto k : keys)
			REQUIRE(visited.count(k) == 1);

		// calling with a non-existent key should return first key
		uint32_t bogus = 9999;
		REQUIRE(lru_map->map_get_next_key(&bogus, &next_key) == 0);
		REQUIRE(visited.count(next_key) == 1);
	}

	// Cleanup
	if (lru_map) {
		shm.destroy_ptr(lru_map);
	}
}
