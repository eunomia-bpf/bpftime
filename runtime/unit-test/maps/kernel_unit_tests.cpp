#include "bpf_map/userspace/array_map.hpp"
#include "bpf_map/userspace/per_cpu_array_map.hpp"
#include "bpf_map/userspace/per_cpu_hash_map.hpp"
#include "catch2/catch_test_macros.hpp"
#include "linux/bpf.h"
#include "spdlog/spdlog.h"
#include "unit-test/common_def.hpp"
#include <algorithm>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include "bpf_map/userspace/var_hash_map.hpp"
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <pthread.h>
#include <thread>
#include <unistd.h>
#include <vector>
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

static void test_hashmap(std::string memory_name = SHM_NAME,
			 bool should_create_shm = true)
{
	shm_initializer shm(memory_name.c_str(), should_create_shm);
	auto &mem = *shm.mem;

	long long key, next_key, first_key, value;
	var_size_hash_map_impl map(mem, sizeof(key), sizeof(value), 2, 0);
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
TEST_CASE("test_hashmap (kernel)", "[kernel]")
{
	test_hashmap();
}
static void test_hashmap_sizes(std::string memory_name = SHM_NAME,
			       bool should_create_shm = true)
{
	shm_initializer shm(memory_name.c_str(), should_create_shm);
	auto &mem = *shm.mem;
	int i, j;

	for (i = 1; i <= 512; i <<= 1)
		for (j = 1; j <= 1 << 18; j <<= 1) {
			var_size_hash_map_impl map(mem, i, j, 2, 0);
			usleep(10);
		}
}
TEST_CASE("test_hashmap_sizes (kernel)", "[kernel]")
{
	test_hashmap_sizes();
}

/** Macros from kernel source */
#define __bpf_percpu_val_align __attribute__((__aligned__(8)))

#define BPF_DECLARE_PERCPU(type, name)                                         \
	struct {                                                               \
		type v; /* padding */                                          \
	} __bpf_percpu_val_align name[sysconf(_SC_NPROCESSORS_ONLN)]
#define bpf_percpu(name, cpu) name[(cpu)].v

static void test_hashmap_percpu(std::string memory_name = SHM_NAME,
				bool should_create_shm = true)
{
	shm_initializer shm(memory_name.c_str(), should_create_shm);
	auto &mem = *shm.mem;

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

TEST_CASE("test_hashmap_percpu (kernel)", "[kernel]")
{
	test_hashmap_percpu();
}

#define VALUE_SIZE 3
static inline var_size_hash_map_impl
helper_fill_hashmap(int max_entries, managed_shared_memory &mem)
{
	int i, ret;
	long long key, value[VALUE_SIZE] = {};

	var_size_hash_map_impl map(mem, sizeof(key), sizeof(value), max_entries,
				   0);

	for (i = 0; i < max_entries; i++) {
		key = i;
		value[0] = key;
		ret = map.elem_update(&key, value, BPF_NOEXIST);
		REQUIRE((ret == 0 && "Unable to update hash map"));
	}

	return map;
}

static void test_hashmap_walk(std::string memory_name = SHM_NAME,
			      bool should_create_shm = true)
{
	shm_initializer shm(memory_name.c_str(), should_create_shm);
	auto &mem = *shm.mem;
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

TEST_CASE("test_hashmap_walk (kernel)", "[kernel]")
{
	test_hashmap_walk();
}

static void test_arraymap(std::string memory_name = SHM_NAME,
			  bool should_create_shm = true)
{
	shm_initializer shm(memory_name.c_str(), should_create_shm);
	auto &mem = *shm.mem;
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

TEST_CASE("test_arraymap (kernel)", "[kernel]")
{
	test_arraymap();
}
static void test_arraymap_percpu(std::string memory_name = SHM_NAME,
				 bool should_create_shm = true)
{
	shm_initializer shm(memory_name.c_str(), should_create_shm);
	auto &mem = *shm.mem;
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
TEST_CASE("test_arraymap_percpu (kernel)", "[kernel]")
{
	test_arraymap_percpu();
}
TEST_CASE("test_arraymap_percpu_many_keys (kernel)", "[kernel]")
{
	shm_initializer shm(SHM_NAME, true);
	auto &mem = *shm.mem;
	unsigned int nr_cpus = sysconf(_SC_NPROCESSORS_ONLN);
	BPF_DECLARE_PERCPU(long, values);
	/* nr_keys is not too large otherwise the test stresses percpu
	 * allocator more than anything else
	 */
	unsigned int nr_keys = 2000;
	int key, i;

	per_cpu_array_map_impl map(mem, sizeof(bpf_percpu(values, 0)), nr_keys);

	for (i = 0; i < (int)nr_cpus; i++)
		bpf_percpu(values, i) = i + 10;

	for (key = 0; key < (int)nr_keys; key++)
		REQUIRE(map.elem_update_userspace(&key, values, BPF_ANY) == 0);

	for (key = 0; key < (int)nr_keys; key++) {
		for (i = 0; i < (int)nr_cpus; i++)
			bpf_percpu(values, i) = 0;
		void *value_ptr;
		REQUIRE((value_ptr = map.elem_lookup_userspace(&key)) !=
			nullptr);
		memcpy(values, value_ptr, sizeof(values));
		for (i = 0; i < (int)nr_cpus; i++)
			REQUIRE(bpf_percpu(values, i) == i + 10);
	}
}
#define MAP_SIZE (32 * 1024)
TEST_CASE("test_map_large (kernel)", "[kernel][large_memory]")
{
	shm_initializer shm(SHM_NAME, true, 150 << 20);
	auto &mem = *shm.mem;
	struct bigkey {
		int a;
		char b[4096];
		long long c;
	} key;
	int i, value;

	var_size_hash_map_impl map(mem, sizeof(key), sizeof(value), MAP_SIZE,
				   0);
	const auto lookup_helper = [&](const void *key, void *value) -> long {
		auto returned_value = map.elem_lookup(key);
		if (!returned_value)
			return -1;
		memcpy(value, returned_value, map.get_value_size());
		return 0;
	};

	for (i = 0; i < MAP_SIZE; i++) {
		key = (struct bigkey){ .c = i };
		value = i;

		REQUIRE(map.elem_update(&key, &value, BPF_NOEXIST) == 0);
	}

	key.c = -1;
	REQUIRE((map.elem_update(&key, &value, BPF_NOEXIST) < 0 &&
		 errno == E2BIG));

	/* Iterate through all elements. */
	REQUIRE(map.map_get_next_key(NULL, &key) == 0);
	key.c = -1;
	for (i = 0; i < MAP_SIZE; i++)
		REQUIRE(map.map_get_next_key(&key, &key) == 0);
	REQUIRE((map.map_get_next_key(&key, &key) < 0 && errno == ENOENT));

	key.c = 0;
	REQUIRE((lookup_helper(&key, &value) == 0 && value == 0));
	key.a = 1;
	REQUIRE((lookup_helper(&key, &value) < 0 && errno == ENOENT));
}

static void
run_parallel(int tasks, std::function<void(int idx)> fn,
	     std::optional<std::unique_ptr<shm_initializer> > common_shm = {})
{
	std::vector<std::thread> thds;
	for (int i = 0; i < tasks; i++) {
		thds.push_back(std::thread(fn, i));
	}
	for (auto &thd : thds)
		thd.join();
}

TEST_CASE("test_map_stress (kernel)", "[kernel]")
{
	const auto start_and_run_task = [&](void (*fn)(std::string, bool),
					    const char *shm_name) {
		run_parallel(
			100,
			[=](int idx) {
				SPDLOG_DEBUG(
					"Started testing thread with memory name {}, index {}",
					shm_name, idx);
				fn(shm_name, false);
			},
			std::make_unique<shm_initializer>(shm_name, true));
	};

	start_and_run_task(test_hashmap, "_SHM__PARALLEL_TEST_HASHMAP");
	start_and_run_task(test_hashmap_percpu,
			   "_SHM__PARALLEL_TEST_HASHMAP_PERCPU");
	start_and_run_task(test_hashmap_sizes,
			   "_SHM__PARALLEL_TEST_HASHMAP_SIZES");

	start_and_run_task(test_arraymap, "_SHM__PARALLEL_TEST_ARRAYMAP");
	start_and_run_task(test_arraymap_percpu,
			   "_SHM__PARALLEL_TEST_ARRAYMAP_PERCPU");
}
#define MAP_RETRIES 20
#define MAX_DELAY_US 50000
#define MIN_DELAY_RANGE_US 5000
#define TASKS 20 /* Kernel uses 100 here, but it will be too slow for us...*/
#define DO_UPDATE 1
#define DO_DELETE 0

struct pthread_spinlock_guard {
	pthread_spinlock_t &lock;
	pthread_spinlock_guard(pthread_spinlock_t &lock) : lock(lock)
	{
		pthread_spin_lock(&lock);
	}
	~pthread_spinlock_guard()
	{
		pthread_spin_unlock(&lock);
	}
};

static int map_update_retriable(var_size_hash_map_impl &map, const void *key,
				const void *value, int flags, int attempts,
				pthread_spinlock_t &lock)
{
	int delay = rand() % MIN_DELAY_RANGE_US;

	while (true) {
		{
			pthread_spinlock_guard guard(lock);
			if (map.elem_update(key, value, flags) == 0)
				break;
		}
		if (!attempts || (errno != EAGAIN && errno != EBUSY))
			return -errno;

		if (delay <= MAX_DELAY_US / 2)
			delay *= 2;

		usleep(delay);
		attempts--;
	}

	return 0;
}

static int map_delete_retriable(var_size_hash_map_impl &map, const void *key,
				int attempts, pthread_spinlock_t &lock)
{
	int delay = rand() % MIN_DELAY_RANGE_US;

	while (true) {
		{
			pthread_spinlock_guard guard(lock);
			if (map.elem_delete(key) == 0)
				break;
		}
		if (!attempts || (errno != EAGAIN && errno != EBUSY))
			return -errno;

		if (delay <= MAX_DELAY_US / 2)
			delay *= 2;

		usleep(delay);
		attempts--;
	}

	return 0;
}

static void test_update_delete(unsigned int fn, int do_update,
			       var_size_hash_map_impl &map,
			       const char *shm_name, pthread_spinlock_t &lock)
{
	int i, key, value, err;

	if (fn & 1)
		test_hashmap_walk(shm_name, false);
	for (i = fn; i < MAP_SIZE; i += TASKS) {
		key = value = i;

		if (do_update) {
			err = map_update_retriable(map, &key, &value,
						   BPF_NOEXIST, MAP_RETRIES,
						   lock);
			if (err)
				SPDLOG_ERROR("error {} {}\n", err, errno);
			REQUIRE(err == 0);
			err = map_update_retriable(map, &key, &value, BPF_EXIST,
						   MAP_RETRIES, lock);
			if (err)
				SPDLOG_ERROR("error {} {}\n", err, errno);
			REQUIRE(err == 0);
		} else {
			err = map_delete_retriable(map, &key, MAP_RETRIES,
						   lock);
			if (err)
				SPDLOG_ERROR("error {} {}\n", err, errno);
			REQUIRE(err == 0);
		}
	}
}

TEST_CASE("test_map_parallel (kernel)", "[kernel][large_memory]")
{
	shm_initializer shm(SHM_NAME, true, 150 << 20);
	auto &mem = *shm.mem;

	int i, key = 0, value = 0, j = 0;

	var_size_hash_map_impl map(mem, sizeof(key), sizeof(value), MAP_SIZE,
				   0);
	pthread_spinlock_t lock;
	pthread_spin_init(&lock, 0);
again:
	/* Use the same fd in children to add elements to this map:
	 * child_0 adds key=0, key=1024, key=2048, ...
	 * child_1 adds key=1, key=1025, key=2049, ...
	 * child_1023 adds key=1023, ...
	 */
	// data[0] = fd;
	// data[1] = DO_UPDATE;
	// run_parallel(TASKS, test_update_delete, data);
	run_parallel(TASKS, [&](int idx) {
		test_update_delete(idx, DO_UPDATE, map, SHM_NAME, lock);
	});
	SPDLOG_DEBUG("parallel update done");
	/* Check that key=0 is already there. */
	REQUIRE((map.elem_update(&key, &value, BPF_NOEXIST) < 0 &&
		 errno == EEXIST));

	/* Check that all elements were inserted. */
	REQUIRE(map.map_get_next_key(NULL, &key) == 0);
	key = -1;
	for (i = 0; i < MAP_SIZE; i++)
		REQUIRE(map.map_get_next_key(&key, &key) == 0);
	REQUIRE((map.map_get_next_key(&key, &key) < 0 && errno == ENOENT));

	/* Another check for all elements */
	for (i = 0; i < MAP_SIZE; i++) {
		key = MAP_SIZE - i - 1;
		int *ptr = (int *)map.elem_lookup(&key);
		REQUIRE(ptr != nullptr);
		value = *ptr;
		REQUIRE(value == key);
	}

	/* Now let's delete all elemenets in parallel. */
	// data[1] = DO_DELETE;
	run_parallel(TASKS, [&](int idx) {
		test_update_delete(idx, DO_DELETE, map, SHM_NAME, lock);
	});

	SPDLOG_DEBUG("parallel delete done");
	/* Nothing should be left. */
	key = -1;
	REQUIRE((map.map_get_next_key(NULL, &key) < 0 && errno == ENOENT));
	REQUIRE((map.map_get_next_key(&key, &key) < 0 && errno == ENOENT));

	key = 0;
	map.elem_delete(&key);
	if (j++ < 5)
		goto again;
	pthread_spin_destroy(&lock);
}

TEST_CASE("test_map_rdonly (kernel)", "[kernel]")
{
	shm_initializer shm(SHM_NAME, true);
	auto &mem = *shm.mem;
	int key = 0, value = 0;

	var_size_hash_map_impl map(mem, sizeof(key), sizeof(value), MAP_SIZE,
				   BPF_F_RDONLY);
	key = 1;
	value = 1234;
	/* Try to insert key=1 element. */
	REQUIRE((map.elem_update(&key, &value, BPF_ANY) < 0 && errno == EPERM));

	/* Check that key=1 is not found. */
	REQUIRE((map.elem_lookup(&key) == nullptr && errno == ENOENT));
	REQUIRE((map.map_get_next_key(&key, &value) < 0 && errno == ENOENT));
}

TEST_CASE("test_map_wronly (kernel)", "[kernel]")
{
	shm_initializer shm(SHM_NAME, true);
	auto &mem = *shm.mem;
	int key = 0, value = 0;

	var_size_hash_map_impl map(mem, sizeof(key), sizeof(value), MAP_SIZE,
				   BPF_F_WRONLY);
	key = 1;
	value = 1234;
	/* Insert key=1 element. */
	REQUIRE(map.elem_update(&key, &value, BPF_ANY) == 0);

	/* Check that reading elements and keys from the map is not allowed. */
	REQUIRE((map.elem_lookup(&key) == nullptr && errno == EPERM));
	REQUIRE((map.map_get_next_key(&key, &value) < 0 && errno == EPERM));
}
