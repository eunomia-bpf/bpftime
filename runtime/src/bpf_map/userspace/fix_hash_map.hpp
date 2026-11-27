/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _FIX_HASHMAP_HPP
#define _FIX_HASHMAP_HPP
#include <boost/interprocess/managed_shared_memory.hpp>
#include <bpf_map/map_common_def.hpp>
#include <bpf_map/bpftime_hash_map.hpp>

namespace bpftime
{

using namespace boost::interprocess;

class fix_size_hash_map_impl {
	bpftime_hash_map map_impl;
	uint32_t _value_size;

    public:
	const static bool should_lock = false;
	fix_size_hash_map_impl(managed_shared_memory &memory,
			       size_t num_buckets, size_t key_size,
			       size_t value_size);

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags,
			 bool from_syscall = true);

	long elem_delete(const void *key, bool from_syscall = true);

	int lookup_and_delete(const void *key, void *value_out,
			      bool from_syscall = false);

	int map_get_next_key(const void *key, void *next_key);
};

} // namespace bpftime
#endif
