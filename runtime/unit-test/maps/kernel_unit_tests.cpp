#include "bpf_map/userspace/array_map.hpp"
#include "bpf_map/userspace/per_cpu_array_map.hpp"
#include "bpf_map/userspace/per_cpu_hash_map.hpp"
#include "catch2/catch_test_macros.hpp"
#include "linux/bpf.h"
#include "unit-test/common_def.hpp"
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include "bpf_map/userspace/var_hash_map.hpp"
#include <cstring>
#include <unistd.h>
static const char *SHM_NAME = "_HASH_MAP_TEST";

using namespace boost::interprocess;
using namespace bpftime;

TEST_CASE("Test hash map (kernel)")
{
	shm_remove remover(SHM_NAME);
	managed_shared_memory mem(boost::interprocess::create_only, SHM_NAME,
				  20 << 20);

	long long key, next_key, first_key, value;
	var_size_hash_map_impl map(mem, sizeof(key), sizeof(value), 2);
	key = 1;
	value = 1234;
	/* Insert key=1 element. */
	REQUIRE(map.elem_update(&key, &value, BPF_ANY) == 0);

	value = 0;
	/* BPF_NOEXIST means add new element if it doesn't exist. */
	REQUIRE(map.elem_update(&key, &value, BPF_NOEXIST) < 0);
	REQUIRE(/* key=1 already exists. */ errno == EEXIST);

	/* -1 is an invalid flag. */
	REQUIRE(map.elem_update(&key, &value, -1) < 0);
	REQUIRE(errno == EINVAL);

	/* Check that key=1 can be found. */
	{
		auto ptr = map.elem_lookup(&key);
		REQUIRE(ptr != nullptr);
		REQUIRE(*(long long *)ptr == 1234);
	}

	key = 2;
	value = 1234;
	/* Insert key=2 element. */
	REQUIRE(map.elem_update(&key, &value, BPF_ANY) == 0);

	/* Check that key=2 matches the value and delete it */
	REQUIRE(map.lookup_and_delete(&key, &value) == 0);
	REQUIRE(value == 1234);

	/* Check that key=2 is not found. */
	REQUIRE(map.elem_lookup(&key) == nullptr);
	REQUIRE(errno == ENOENT);

	/* BPF_EXIST means update existing element. */
	REQUIRE(map.elem_update(&key, &value, BPF_EXIST) < 0);

	REQUIRE(/* key=2 is not there. */
		errno == ENOENT);

	/* Insert key=2 element. */
	REQUIRE(map.elem_update(&key, &value, BPF_NOEXIST) == 0);

	/* key=1 and key=2 were inserted, check that key=0 cannot be
	 * inserted due to max_entries limit.
	 */
	key = 0;
	REQUIRE(map.elem_update(&key, &value, BPF_NOEXIST) < 0);
	REQUIRE(errno == E2BIG);

	/* Update existing element, though the map is full. */
	key = 1;
	REQUIRE(map.elem_update(&key, &value, BPF_EXIST) == 0);
	key = 2;
	REQUIRE(map.elem_update(&key, &value, BPF_ANY) == 0);
	key = 3;
	REQUIRE(map.elem_update(&key, &value, BPF_NOEXIST) < 0);
	REQUIRE(errno == E2BIG);

	/* Check that key = 0 doesn't exist. */
	key = 0;
	REQUIRE(map.elem_delete(&key) < 0);
	REQUIRE(errno == ENOENT);
	/* Iterate over two elements. */
	REQUIRE(map.map_get_next_key(NULL, &first_key) == 0);
	REQUIRE((first_key == 1 || first_key == 2));
	REQUIRE(map.map_get_next_key(&key, &next_key) == 0);
	REQUIRE((next_key == first_key));
	REQUIRE(map.map_get_next_key(&next_key, &next_key) == 0);
	REQUIRE(((next_key == 1 || next_key == 2) && (next_key != first_key)));
	REQUIRE(map.map_get_next_key(&next_key, &next_key) < 0);
	REQUIRE(errno == ENOENT);

	/* Delete both elements. */
	key = 1;
	REQUIRE(map.elem_delete(&key) == 0);
	key = 2;
	REQUIRE(map.elem_delete(&key) == 0);
	REQUIRE(map.elem_delete(&key) < 0);
	REQUIRE(errno == ENOENT);

	key = 0;
	/* Check that map is empty. */
	REQUIRE(map.map_get_next_key(NULL, &next_key) < 0);
	REQUIRE(errno == ENOENT);
	REQUIRE(map.map_get_next_key(&key, &next_key) < 0);
	REQUIRE(errno == ENOENT);
}

TEST_CASE("test_hashmap_sizes (kernel)")
{
	shm_remove remover(SHM_NAME);
	managed_shared_memory mem(boost::interprocess::create_only, SHM_NAME,
				  20 << 20);
	int i, j;

	for (i = 1; i <= 512; i <<= 1)
		for (j = 1; j <= 1 << 18; j <<= 1) {
			var_size_hash_map_impl map(mem, i, j, 2);
			usleep(10);
		}
}

/** Macros from kernel source */
#define __bpf_percpu_val_align __attribute__((__aligned__(8)))

#define BPF_DECLARE_PERCPU(type, name)                                         \
	struct {                                                               \
		type v; /* padding */                                          \
	} __bpf_percpu_val_align name[sysconf(_SC_NPROCESSORS_ONLN)]
#define bpf_percpu(name, cpu) name[(cpu)].v

TEST_CASE("test_hashmap_percpu (kernel)")
{
	shm_remove remover(SHM_NAME);
	managed_shared_memory mem(boost::interprocess::create_only, SHM_NAME,
				  20 << 20);

	unsigned int nr_cpus = sysconf(_SC_NPROCESSORS_ONLN);
	BPF_DECLARE_PERCPU(long, value);
	long long key, next_key, first_key;
	int expected_key_mask = 0;
	int i;
	per_cpu_hash_map_impl map(mem, sizeof(key),
				  sizeof(bpf_percpu(value, 0)), 2);

	for (i = 0; i < (int)nr_cpus; i++)
		bpf_percpu(value, i) = i + 100;

	key = 1;
	/* Insert key=1 element. */
	REQUIRE(!(expected_key_mask & key));
	REQUIRE(map.elem_update_userspace(&key, value, BPF_ANY) == 0);

	/* Lookup and delete elem key=1 and check value. */
	REQUIRE((map.lookup_and_delete_userspace(&key, value) == 0 &&
		 bpf_percpu(value, 0) == 100));

	for (i = 0; i < (int)nr_cpus; i++)
		bpf_percpu(value, i) = i + 100;

	/* Insert key=1 element which should not exist. */
	REQUIRE(map.elem_update_userspace(&key, value, BPF_NOEXIST) == 0);
	expected_key_mask |= key;

	/* BPF_NOEXIST means add new element if it doesn't exist. */
	REQUIRE((map.elem_update_userspace(&key, value, BPF_NOEXIST) < 0 &&
		 /* key=1 already exists. */
		 errno == EEXIST));

	/* -1 is an invalid flag. */
	REQUIRE((map.elem_update(&key, value, -1) < 0 && errno == EINVAL));

	const auto lookup_helper = [&](const void *key, void *value) -> long {
		auto returned_value = map.elem_lookup_userspace(key);
		if (!returned_value)
			return -1;
		memcpy(value, returned_value,
		       map.get_value_size() * map.getncpu());
		return 0;
	};

	/* Check that key=1 can be found. Value could be 0 if the lookup
	 * was run from a different CPU.
	 */
	bpf_percpu(value, 0) = 1;
	REQUIRE((lookup_helper(&key, value) == 0 &&
		 bpf_percpu(value, 0) == 100));

	key = 2;
	/* Check that key=2 is not found. */
	REQUIRE((lookup_helper(&key, value) < 0 && errno == ENOENT));

	/* BPF_EXIST means update existing element. */
	REQUIRE((map.elem_update_userspace(&key, value, BPF_EXIST) < 0 &&
		 /* key=2 is not there. */
		 errno == ENOENT));

	/* Insert key=2 element. */
	REQUIRE(!(expected_key_mask & key));
	REQUIRE(map.elem_update_userspace(&key, value, BPF_NOEXIST) == 0);
	expected_key_mask |= key;

	/* key=1 and key=2 were inserted, check that key=0 cannot be
	 * inserted due to max_entries limit.
	 */
	key = 0;
	REQUIRE((map.elem_update_userspace(&key, value, BPF_NOEXIST) < 0 &&
		 errno == E2BIG));

	/* Check that key = 0 doesn't exist. */
	REQUIRE((map.elem_delete_userspace(&key) < 0 && errno == ENOENT));

	/* Iterate over two elements. */
	REQUIRE((map.map_get_next_key(NULL, &first_key) == 0 &&
		 ((expected_key_mask & first_key) == first_key)));
	while (!map.map_get_next_key(&key, &next_key)) {
		if (first_key) {
			REQUIRE(next_key == first_key);
			first_key = 0;
		}
		REQUIRE((expected_key_mask & next_key) == next_key);
		expected_key_mask &= ~next_key;

		REQUIRE(lookup_helper(&next_key, value) == 0);

		for (i = 0; i < (int)nr_cpus; i++)
			REQUIRE(bpf_percpu(value, i) == i + 100);

		key = next_key;
	}
	REQUIRE(errno == ENOENT);

	/* Update with BPF_EXIST. */
	key = 1;
	REQUIRE(map.elem_update_userspace(&key, value, BPF_EXIST) == 0);

	/* Delete both elements. */
	key = 1;
	REQUIRE(map.elem_delete_userspace(&key) == 0);
	key = 2;
	REQUIRE(map.elem_delete_userspace(&key) == 0);
	REQUIRE((map.elem_delete_userspace(&key) < 0 && errno == ENOENT));

	key = 0;
	/* Check that map is empty. */
	REQUIRE((map.map_get_next_key(NULL, &next_key) < 0 && errno == ENOENT));
	REQUIRE((map.map_get_next_key(&key, &next_key) < 0 && errno == ENOENT));
}

#define VALUE_SIZE 3
static inline var_size_hash_map_impl
helper_fill_hashmap(int max_entries, managed_shared_memory &mem)
{
	int i, fd, ret;
	long long key, value[VALUE_SIZE] = {};

	var_size_hash_map_impl map(mem, sizeof(key), sizeof(value),
				   max_entries);

	for (i = 0; i < max_entries; i++) {
		key = i;
		value[0] = key;
		ret = map.elem_update(&key, value, BPF_NOEXIST);
		REQUIRE((ret == 0 && "Unable to update hash map"));
	}

	return map;
}

TEST_CASE("test_hashmap_walk (kernel)")
{
	shm_remove remover(SHM_NAME);
	managed_shared_memory mem(boost::interprocess::create_only, SHM_NAME,
				  20 << 20);
	int i, max_entries = 10000;
	long long key, value[VALUE_SIZE], next_key;
	bool next_key_valid = true;

	auto map = helper_fill_hashmap(max_entries, mem);

	const auto lookup_helper = [&](const void *key, void *value) -> long {
		auto returned_value = map.elem_lookup(key);
		if (!returned_value)
			return -1;
		memcpy(value, returned_value, map.get_value_size());
		return 0;
	};

	for (i = 0; map.map_get_next_key(!i ? NULL : &key, &next_key) == 0;
	     i++) {
		key = next_key;
		REQUIRE(lookup_helper(&key, value) == 0);
	}

	REQUIRE(i == max_entries);

	REQUIRE(map.map_get_next_key(NULL, &key) == 0);
	for (i = 0; next_key_valid; i++) {
		next_key_valid = map.map_get_next_key(&key, &next_key) == 0;
		REQUIRE(lookup_helper(&key, value) == 0);
		value[0]++;
		REQUIRE(map.elem_update(&key, value, BPF_EXIST) == 0);
		key = next_key;
	}

	REQUIRE(i == max_entries);

	for (i = 0; map.map_get_next_key(!i ? NULL : &key, &next_key) == 0;
	     i++) {
		key = next_key;
		REQUIRE(lookup_helper(&key, value) == 0);
		REQUIRE(value[0] - 1 == key);
	}

	REQUIRE(i == max_entries);
}

TEST_CASE("test_arraymap (kernel)")
{
	shm_remove remover(SHM_NAME);
	managed_shared_memory mem(boost::interprocess::create_only, SHM_NAME,
				  20 << 20);
	int key, next_key;
	long long value;
	auto value_size = sizeof(value);
	array_map_impl map(mem, sizeof(value), 2);
	const auto lookup_helper = [&](const void *key, void *value) -> long {
		auto returned_value = map.elem_lookup(key);
		if (!returned_value)
			return -1;
		memcpy(value, returned_value, sizeof(value_size));
		return 0;
	};

	key = 1;
	value = 1234;
	/* Insert key=1 element. */
	REQUIRE(map.elem_update(&key, &value, BPF_ANY) == 0);

	value = 0;
	REQUIRE((map.elem_update(&key, &value, BPF_NOEXIST) < 0 &&
		 errno == EEXIST));

	/* Check that key=1 can be found. */
	REQUIRE((lookup_helper(&key, &value) == 0 && value == 1234));

	key = 0;
	/* Check that key=0 is also found and zero initialized. */
	REQUIRE((lookup_helper(&key, &value) == 0 && value == 0));

	/* key=0 and key=1 were inserted, check that key=2 cannot be inserted
	 * due to max_entries limit.
	 */
	key = 2;
	REQUIRE((map.elem_update(&key, &value, BPF_EXIST) < 0 &&
		 errno == E2BIG));

	/* Check that key = 2 doesn't exist. */
	REQUIRE((lookup_helper(&key, &value) < 0 && errno == ENOENT));

	/* Iterate over two elements. */
	REQUIRE((map.map_get_next_key(NULL, &next_key) == 0 && next_key == 0));
	REQUIRE((map.map_get_next_key(&key, &next_key) == 0 && next_key == 0));
	REQUIRE((map.map_get_next_key(&next_key, &next_key) == 0 &&
		 next_key == 1));
	REQUIRE((map.map_get_next_key(&next_key, &next_key) < 0 &&
		 errno == ENOENT));

	/* Delete shouldn't succeed. */
	key = 1;
	REQUIRE((map.elem_delete(&key) < 0 && errno == EINVAL));
}

TEST_CASE("test_arraymap_percpu (kernel)")
{
	shm_remove remover(SHM_NAME);
	managed_shared_memory mem(boost::interprocess::create_only, SHM_NAME,
				  20 << 20);
	unsigned int nr_cpus = sysconf(_SC_NPROCESSORS_ONLN);
	BPF_DECLARE_PERCPU(long, values);
	int key, next_key, i;
	per_cpu_array_map_impl map(mem, sizeof(bpf_percpu(values, 0)), 2);
	const auto lookup_helper = [&](const void *key, void *value) -> long {
		auto returned_value = map.elem_lookup_userspace(key);
		if (!returned_value)
			return -1;
		memcpy(value, returned_value, sizeof(values));
		return 0;
	};

	for (i = 0; i < (int)nr_cpus; i++)
		bpf_percpu(values, i) = i + 100;

	key = 1;
	/* Insert key=1 element. */
	REQUIRE(map.elem_update_userspace(&key, values, BPF_ANY) == 0);

	bpf_percpu(values, 0) = 0;
	REQUIRE((map.elem_update_userspace(&key, values, BPF_NOEXIST) < 0 &&
		 errno == EEXIST));

	/* Check that key=1 can be found. */
	REQUIRE((lookup_helper(&key, values) == 0 &&
		 bpf_percpu(values, 0) == 100));

	key = 0;
	/* Check that key=0 is also found and zero initialized. */
	REQUIRE((lookup_helper(&key, values) == 0 &&
		 bpf_percpu(values, 0) == 0 &&
		 bpf_percpu(values, nr_cpus - 1) == 0));

	/* Check that key=2 cannot be inserted due to max_entries limit. */
	key = 2;
	REQUIRE((map.elem_update_userspace(&key, values, BPF_EXIST) < 0 &&
		 errno == E2BIG));

	/* Check that key = 2 doesn't exist. */
	REQUIRE((lookup_helper(&key, values) < 0 && errno == ENOENT));

	/* Iterate over two elements. */
	REQUIRE((map.map_get_next_key(NULL, &next_key) == 0 && next_key == 0));
	REQUIRE((map.map_get_next_key(&key, &next_key) == 0 && next_key == 0));
	REQUIRE((map.map_get_next_key(&next_key, &next_key) == 0 &&
		 next_key == 1));
	REQUIRE((map.map_get_next_key(&next_key, &next_key) < 0 &&
		 errno == ENOENT));

	/* Delete shouldn't succeed. */
	key = 1;
	REQUIRE((map.elem_delete_userspace(&key) < 0 && errno == EINVAL));
}
