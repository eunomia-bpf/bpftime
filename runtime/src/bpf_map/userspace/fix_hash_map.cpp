/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "spdlog/spdlog.h"
#include <bpf_map/userspace/fix_hash_map.hpp>
#include <algorithm>
#include <functional>
#include <unistd.h>
#include <unordered_map>

namespace bpftime
{

fix_size_hash_map_impl::fix_size_hash_map_impl(managed_shared_memory &memory,
				      size_t num_buckets, size_t key_size,
				      size_t value_size)
	: map_impl(memory, num_buckets, key_size, value_size),
	  _key_size(key_size), _value_size(value_size),
	  _num_buckets(num_buckets)
{
}

void *fix_size_hash_map_impl::elem_lookup(const void *key)
{
	SPDLOG_TRACE("Peform elem lookup of hash map");
	// Since we use lock here, we don't need to allocate key_vec and
	// value_vec
	return map_impl.elem_lookup(key);
}

long fix_size_hash_map_impl::elem_update(const void *key, const void *value,
					 uint64_t flags)
{
	map_impl.elem_update(key, value);
	return 0;
}

long fix_size_hash_map_impl::elem_delete(const void *key)
{
	map_impl.elem_delete(key);
	return 0;
}

int fix_size_hash_map_impl::map_get_next_key(const void *key, void *next_key)
{
	SPDLOG_TRACE("Peform map get next key of hash map");
	if (next_key == nullptr) {
		errno = EINVAL;
		return -1;
	}
	if (key == nullptr) {
		// get the first key
		for (size_t i = 0; i < _num_buckets; i++) {
			if (map_impl.is_empty(i)) {
				continue;
			}
			memcpy(next_key, map_impl.get_key(i), _key_size);
			return 0;
		}
		errno = ENOENT;
		// no key found
		return -1;
	}
	// get the next key
	void *value_ptr = elem_lookup(key);
	if (value_ptr == nullptr) {
		return map_get_next_key(nullptr, next_key);
	}
	// get_next_key
	size_t index = map_impl.get_index_of_value(value_ptr);
	for (size_t i = index + 1; i < _num_buckets; i++) {
		if (map_impl.is_empty(i)) {
			continue;
		}
		memcpy(next_key, map_impl.get_key(i), _key_size);
		return 0;
	}
	// if this is the last key
	errno = ENOENT;
	return -1;
}

} // namespace bpftime
