/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */

#ifndef _BPFTIME_LRU_FIX_HASH_MAP_HPP
#define _BPFTIME_LRU_FIX_HASH_MAP_HPP

#include "bpf_map/map_common_def.hpp"
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/smart_ptr/shared_ptr.hpp>
#include <boost/interprocess/smart_ptr/weak_ptr.hpp>
#include <optional>
#include <boost/interprocess/containers/map.hpp>
#include <boost/container_hash/hash_fwd.hpp>
#include <boost/unordered/unordered_map.hpp>
#include <boost/functional/hash.hpp>
namespace bpftime
{
using vec_allocator = boost::interprocess::allocator<
	char, boost::interprocess::managed_shared_memory::segment_manager>;
using lru_linklist_entry_shared_ptr = boost::interprocess::managed_shared_ptr<
	struct lru_linklist_entry,
	boost::interprocess::managed_shared_memory::segment_manager>::type;
using lru_linklist_entry_weak_ptr = boost::interprocess::managed_weak_ptr<
	struct lru_linklist_entry,
	boost::interprocess::managed_shared_memory::segment_manager>::type;

struct lru_linklist_entry {
	bytes_vec key;
	lru_linklist_entry_shared_ptr next_entry;
	lru_linklist_entry_weak_ptr previous_entry;
	lru_linklist_entry(const bytes_vec &key) : key(key)
	{
	}
};

struct hash_map_value {
	bytes_vec value;
	lru_linklist_entry_shared_ptr linked_list_entry;
};

class lru_var_hash_map_impl {
	using bi_map_value_ty = std::pair<const bytes_vec, hash_map_value>;
	using bi_map_allocator = boost::interprocess::allocator<
		bi_map_value_ty,
		boost::interprocess::managed_shared_memory::segment_manager>;
	using shm_hash_map =
		boost::unordered_map<bytes_vec, hash_map_value, bytes_vec_hasher,
				     std::equal_to<bytes_vec>, bi_map_allocator>;
	shm_hash_map map_impl;
	lru_linklist_entry_shared_ptr lru_link_list_head;
	lru_linklist_entry_shared_ptr lru_link_list_tail;

	size_t key_size;
	size_t value_size;
	size_t max_entries;
	bytes_vec key_vec;
	bytes_vec value_vec;

	boost::interprocess::managed_shared_memory &memory;

	void move_to_head(lru_linklist_entry_shared_ptr entry);
	lru_linklist_entry_shared_ptr insert_new_entry(const bytes_vec &key);
	void evict_entry(lru_linklist_entry_shared_ptr entry);

    public:
	const static bool should_lock = true;
	lru_var_hash_map_impl(boost::interprocess::managed_shared_memory &memory,
			      size_t key_size, size_t value_size,
			      size_t max_entries);
	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);
};
} // namespace bpftime

#endif
