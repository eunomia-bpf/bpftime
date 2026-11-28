/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026, eunomia-bpf org
 * All rights reserved.
 *
 * The BPF-program-visible map helper return code. The kernel's
 * bpf_map_*_elem helpers return 0 or a negative errno value, and programs
 * branch on -EEXIST/-ENOENT/-E2BIG, so the adapter must translate the
 * backends' -1 + host-errno convention at the helper boundary. The
 * syscall-facing shims keep -1 + errno and are covered by
 * test_map_syscall_semantics.cpp.
 */
#include "catch2/catch_test_macros.hpp"
#include "linux/bpf.h"
#include "unit-test/common_def.hpp"
#include "bpftime_shm.hpp"
#include <cerrno>
#include <cstdint>

extern "C" {
uint64_t bpftime_map_update_elem_helper(uint64_t map, uint64_t key,
					uint64_t value, uint64_t flags,
					uint64_t);
uint64_t bpftime_map_delete_elem_helper(uint64_t map, uint64_t key,
					uint64_t flags, uint64_t, uint64_t);
uint64_t bpftime_map_lookup_elem_helper(uint64_t map, uint64_t key,
					 uint64_t, uint64_t, uint64_t);
uint64_t bpftime_map_push_elem_helper(uint64_t map, uint64_t value,
				      uint64_t flags, uint64_t, uint64_t);
uint64_t bpftime_map_pop_elem_helper(uint64_t map, uint64_t value, uint64_t,
				     uint64_t, uint64_t);
uint64_t bpftime_map_peek_elem_helper(uint64_t map, uint64_t value, uint64_t,
				      uint64_t, uint64_t);
}

using namespace boost::interprocess;
using namespace bpftime;

namespace
{
int64_t as_signed(uint64_t value)
{
	return (int64_t)value;
}
} // namespace

#if __linux__
TEST_CASE("map helpers report the kernel negative errno", "[map_helpers]")
{
	::shm_remove remover("_MAP_HELPER_ERRNO_SHM");
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);

	constexpr int lru_hash_fd = 1401;
	REQUIRE(bpftime_maps_create(
			lru_hash_fd, "helper_errno_lru",
			bpf_map_attr{
				.type = (int)bpftime::bpf_map_type::
					BPF_MAP_TYPE_LRU_HASH,
				.key_size = sizeof(uint32_t),
				.value_size = sizeof(uint64_t),
				.max_ents = 2,
				.flags = 0 }) >= 0);

	uint32_t key = 7;
	uint64_t value = 0x1122334455667788ULL;
	const uint64_t key_arg = (uint64_t)(uintptr_t)&key;
	const uint64_t value_arg = (uint64_t)(uintptr_t)&value;

	// Success is 0, not an errno.
	REQUIRE(bpftime_map_update_elem_helper(lru_hash_fd, key_arg, value_arg,
					       BPF_ANY, 0) == 0);

	// BPF_NOEXIST on an existing key: -EEXIST, the value a program tests
	// for to fall back to an update.
	REQUIRE(as_signed(bpftime_map_update_elem_helper(
			lru_hash_fd, key_arg, value_arg, BPF_NOEXIST, 0)) ==
		-EEXIST);

	// BPF_EXIST on an absent key: -ENOENT.
	uint32_t absent_key = 8;
	REQUIRE(as_signed(bpftime_map_update_elem_helper(
			lru_hash_fd, (uint64_t)(uintptr_t)&absent_key,
			value_arg, BPF_EXIST, 0)) == -ENOENT);

	// Deleting an absent key: -ENOENT.
	REQUIRE(as_signed(bpftime_map_delete_elem_helper(
			lru_hash_fd, (uint64_t)(uintptr_t)&absent_key, 0, 0,
			0)) == -ENOENT);

	// Deleting a present key succeeds.
	REQUIRE(bpftime_map_delete_elem_helper(lru_hash_fd, key_arg, 0, 0, 0) ==
		0);

	bpftime_destroy_global_shm();
}

// The other half of the contract: a helper runs inside an intercepted host
// function, so a backend's failure errno must not survive the helper's
// return. A host thread that had errno set before the helper (a common case:
// it is set by whatever the program was doing) must still see it after.
TEST_CASE("map helpers leave the host errno untouched", "[map_helpers]")
{
	::shm_remove remover("_MAP_HELPER_HOST_ERRNO_SHM");
	bpftime_initialize_global_shm(
		bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);

	constexpr int hash_fd = 1411;
	REQUIRE(bpftime_maps_create(
			hash_fd, "helper_host_errno_hash",
			bpf_map_attr{
				.type = (int)bpftime::bpf_map_type::
					BPF_MAP_TYPE_LRU_HASH,
				.key_size = sizeof(uint32_t),
				.value_size = sizeof(uint64_t),
				.max_ents = 2,
				.flags = 0 }) >= 0);
	constexpr int queue_fd = 1412;
	REQUIRE(bpftime_maps_create(
			queue_fd, "helper_host_errno_queue",
			bpf_map_attr{
				.type = (int)bpftime::bpf_map_type::
					BPF_MAP_TYPE_QUEUE,
				.key_size = 0,
				.value_size = sizeof(uint64_t),
				.max_ents = 4,
				.flags = 0 }) >= 0);

	uint32_t absent_key = 3;
	uint64_t value = 0;
	const uint64_t key_arg = (uint64_t)(uintptr_t)&absent_key;
	const uint64_t value_arg = (uint64_t)(uintptr_t)&value;
	constexpr int sentinel = EDOM;

	// A lookup miss is the hot case: backends set ENOENT, the program only
	// sees a null return.
	errno = sentinel;
	REQUIRE(bpftime_map_lookup_elem_helper(hash_fd, key_arg, 0, 0, 0) == 0);
	REQUIRE(errno == sentinel);

	// Mutation helpers report their error in the return value too.
	errno = sentinel;
	REQUIRE(as_signed(bpftime_map_update_elem_helper(
			hash_fd, key_arg, value_arg, BPF_EXIST, 0)) == -ENOENT);
	REQUIRE(errno == sentinel);
	errno = sentinel;
	REQUIRE(as_signed(bpftime_map_delete_elem_helper(
			hash_fd, key_arg, 0, 0, 0)) == -ENOENT);
	REQUIRE(errno == sentinel);

	// Queue push/pop/peek set ENOENT/E2BIG on the empty and full cases.
	errno = sentinel;
	REQUIRE(as_signed(bpftime_map_pop_elem_helper(queue_fd, value_arg, 0,
						      0, 0)) == -ENOENT);
	REQUIRE(errno == sentinel);
	errno = sentinel;
	REQUIRE(as_signed(bpftime_map_peek_elem_helper(queue_fd, value_arg, 0,
						       0, 0)) == -ENOENT);
	REQUIRE(errno == sentinel);

	// A successful helper also leaves it alone.
	errno = sentinel;
	REQUIRE(bpftime_map_push_elem_helper(queue_fd, value_arg, 0, 0, 0) == 0);
	REQUIRE(errno == sentinel);

	// The syscall-facing path is the opposite: a program libbpf caller
	// branches on -1 + errno, so there it must survive.
	errno = 0;
	REQUIRE(bpftime_map_lookup_elem_with_origin(hash_fd, &absent_key,
						     true) == nullptr);
	REQUIRE(errno == ENOENT);

	bpftime_destroy_global_shm();
}
#endif
