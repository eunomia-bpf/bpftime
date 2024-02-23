/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_PER_CPU_HASH_MAP_HPP
#define _BPFTIME_PER_CPU_HASH_MAP_HPP
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/smart_ptr/unique_ptr.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <maps/map_common_def.hpp>
#include <boost/unordered/unordered_map.hpp>
#include <cstdint>

namespace bpftime
{
namespace shm_common
{
// Implementation of the userspace per cpu hash map
// It's based on boost::interprocess::unordered_map.
// Each element was corresponded to a value with ncpu*value_size, and access from a certain cpu could only access its slot.
// But access from userspace can read all slots
class per_cpu_hash_map_impl {
	using bi_map_value_ty = std::pair<const bytes_vec, bytes_vec>;
	using bi_map_allocator = boost::interprocess::allocator<
		bi_map_value_ty,
		boost::interprocess::managed_shared_memory::segment_manager>;
	using shm_hash_map =
		boost::unordered_map<bytes_vec, bytes_vec, bytes_vec_hasher,
				     std::equal_to<bytes_vec>, bi_map_allocator>;

	shm_hash_map impl;
	uint32_t key_size;
	uint32_t value_size;
	int ncpu;

	bytes_vec key_template, value_template, single_value_template;

    public:
	const static bool should_lock = false;

	per_cpu_hash_map_impl(boost::interprocess::managed_shared_memory &memory,
			      uint32_t key_size, uint32_t value_size);
	per_cpu_hash_map_impl(boost::interprocess::managed_shared_memory &memory,
			      uint32_t key_size, uint32_t value_size, int ncpu);
	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);

	void *elem_lookup_userspace(const void *key);

	long elem_update_userspace(const void *key, const void *value,
				   uint64_t flags);

	long elem_delete_userspace(const void *key);
};
} // namespace shm_common
} // namespace bpftime

#endif
