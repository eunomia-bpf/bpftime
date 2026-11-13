// SPDX-License-Identifier: MIT
//
// Copyright (c) 2026, eunomia-bpf org
// All rights reserved.
//
// Enforcement of BPF_F_RDONLY / BPF_F_WRONLY on the userspace hash map
// backends. The kernel rejects writes (update/delete) to a BPF_F_RDONLY map
// and reads (lookup/get_next_key) from a BPF_F_WRONLY map with EPERM, see
// map_get_sys_perms() checks in kernel/bpf/syscall.c.
#include "catch2/catch_test_macros.hpp"
#include "linux/bpf.h"
#include "unit-test/common_def.hpp"
#include <bpf_map/userspace/var_hash_map.hpp>
#include <bpf_map/userspace/lru_var_hash_map.hpp>
#include <cerrno>
#include <memory>

using namespace boost::interprocess;
using namespace bpftime;

namespace
{

struct shm_initializer {
	std::unique_ptr<shm_remove> remover;
	std::unique_ptr<managed_shared_memory> mem;
	shm_initializer(const char *memory_name, size_t memory_size = 20 << 20)
	{
		remover = std::make_unique<shm_remove>(memory_name);
		mem = std::make_unique<managed_shared_memory>(
		    create_only, memory_name, memory_size);
	}
};

// Both backends expose the same element API and accept the map creation
// flags through their constructor, so the flag matrix is shared.
template <typename MapImpl>
static void run_rdonly_checks(managed_shared_memory &mem)
{
	int key = 1, value = 1234, next_key = 0;
	{
		MapImpl map(mem, sizeof(key), sizeof(value), 16, BPF_F_RDONLY);

		// Every write must fail with EPERM, including updates of keys
		// that do not exist yet. The permission check happens before
		// the BPF_EXIST/NOEXIST/ENOENT/E2BIG logic, so all update
		// flags report EPERM rather than their usual errno.
		errno = 0;
		REQUIRE(map.elem_update(&key, &value, BPF_ANY) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(map.elem_update(&key, &value, BPF_NOEXIST) == -1);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(map.elem_update(&key, &value, BPF_EXIST) == -1);
		REQUIRE(errno == EPERM);

		// Delete is a write as well; on an empty map the old code
		// returned ENOENT, the kernel requires EPERM first.
		errno = 0;
		REQUIRE(map.elem_delete(&key) == -1);
		REQUIRE(errno == EPERM);

		// Reads remain allowed: the map is simply empty.
		errno = 0;
		REQUIRE(map.elem_lookup(&key) == nullptr);
		REQUIRE(errno == ENOENT);
		errno = 0;
		REQUIRE(map.map_get_next_key(nullptr, &next_key) == -1);
		REQUIRE(errno == ENOENT);
	}
}

template <typename MapImpl>
static void run_wronly_checks(managed_shared_memory &mem)
{
	int key = 1, another_key = 2, value = 1234, next_key = 0;
	{
		MapImpl map(mem, sizeof(key), sizeof(value), 16, BPF_F_WRONLY);

		// Writes are allowed on a write-only map.
		REQUIRE(map.elem_update(&key, &value, BPF_ANY) == 0);
		REQUIRE(map.elem_update(&another_key, &value, BPF_NOEXIST) ==
			0);

		// Reads must fail with EPERM.
		errno = 0;
		REQUIRE(map.elem_lookup(&key) == nullptr);
		REQUIRE(errno == EPERM);
		errno = 0;
		REQUIRE(map.map_get_next_key(nullptr, &next_key) == -1);
		REQUIRE(errno == EPERM);

		// Deletes are writes and stay allowed.
		REQUIRE(map.elem_delete(&another_key) == 0);
	}
}

template <typename MapImpl>
static void run_unflagged_checks(managed_shared_memory &mem)
{
	int key = 1, value = 1234, new_value = 5678, next_key = 0;
	{
		MapImpl map(mem, sizeof(key), sizeof(value), 16, 0);

		REQUIRE(map.elem_update(&key, &value, BPF_NOEXIST) == 0);
		// Replacing an existing key must keep working.
		REQUIRE(map.elem_update(&key, &new_value, BPF_EXIST) == 0);

		void *ptr = map.elem_lookup(&key);
		REQUIRE(ptr != nullptr);
		REQUIRE(*static_cast<int *>(ptr) == new_value);

		REQUIRE(map.map_get_next_key(nullptr, &next_key) == 0);
		REQUIRE(next_key == key);

		REQUIRE(map.elem_delete(&key) == 0);
		errno = 0;
		REQUIRE(map.elem_lookup(&key) == nullptr);
		REQUIRE(errno == ENOENT);
	}
}

} // namespace

TEST_CASE("var-size hash map enforces BPF_F_RDONLY/BPF_F_WRONLY",
	  "[map_flags]")
{
	shm_initializer shm("_VAR_HASH_MAP_FLAGS_TEST");
	auto &mem = *shm.mem;
	run_rdonly_checks<var_size_hash_map_impl>(mem);
	run_wronly_checks<var_size_hash_map_impl>(mem);
	run_unflagged_checks<var_size_hash_map_impl>(mem);

	// lookup_and_delete mutates the map, so a read-only map rejects it.
	{
		int key = 1, value = 0;
		var_size_hash_map_impl map(mem, sizeof(key), sizeof(value), 16,
					   BPF_F_RDONLY);
		errno = 0;
		REQUIRE(map.lookup_and_delete(&key, &value) == -1);
		REQUIRE(errno == EPERM);
	}
}

TEST_CASE("LRU hash map enforces BPF_F_RDONLY/BPF_F_WRONLY", "[map_flags]")
{
	shm_initializer shm("_LRU_HASH_MAP_FLAGS_TEST");
	auto &mem = *shm.mem;
	run_rdonly_checks<lru_var_hash_map_impl>(mem);
	run_wronly_checks<lru_var_hash_map_impl>(mem);
	run_unflagged_checks<lru_var_hash_map_impl>(mem);
}
