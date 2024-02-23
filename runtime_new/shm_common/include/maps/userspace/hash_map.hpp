/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _HASHMAP_HPP
#define _HASHMAP_HPP
#include <boost/container_hash/hash_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <maps/map_common_def.hpp>
#include <boost/unordered/unordered_map.hpp>
#include <boost/functional/hash.hpp>

namespace bpftime
{

namespace shm_common
{

// implementation of the userspace hash map
// It's based on a boost::interprocess::unordered_map
class hash_map_impl {
	using bi_map_value_ty = std::pair<const bytes_vec, bytes_vec>;
	using bi_map_allocator = boost::interprocess::allocator<
		bi_map_value_ty,
		boost::interprocess::managed_shared_memory::segment_manager>;
	using shm_hash_map =
		boost::unordered_map<bytes_vec, bytes_vec, bytes_vec_hasher,
				     std::equal_to<bytes_vec>, bi_map_allocator>;
	shm_hash_map map_impl;
	uint32_t _key_size;
	uint32_t _value_size;

	bytes_vec key_vec;
	bytes_vec value_vec;

    public:
	const static bool should_lock = true;
	hash_map_impl(boost::interprocess::managed_shared_memory &memory,
		      uint32_t key_size, uint32_t value_size);

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);
};
} // namespace shm_common

} // namespace bpftime
#endif
