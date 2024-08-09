/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _FIX_HASHMAP_HPP
#define _FIX_HASHMAP_HPP
#include <boost/container_hash/hash_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <bpf_map/map_common_def.hpp>
#include <boost/unordered/unordered_map.hpp>
#include <boost/functional/hash.hpp>
#include <bpf_map/bpftime_hash_map.hpp>

namespace bpftime
{

using namespace boost::interprocess;

// implementation of hash map
class fix_size_hash_map_impl {
	bpftime_hash_map map_impl;
	uint32_t _key_size;
	uint32_t _value_size;
	uint32_t _num_buckets;

    public:
	const static bool should_lock = true;
	fix_size_hash_map_impl(managed_shared_memory &memory, size_t num_buckets,
			     size_t key_size, size_t value_size);

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);
};

} // namespace bpftime
#endif
