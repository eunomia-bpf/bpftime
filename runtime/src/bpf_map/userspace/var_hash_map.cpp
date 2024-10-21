/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "linux/bpf.h"
#include "spdlog/spdlog.h"
#include <bpf_map/userspace/var_hash_map.hpp>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <functional>
#include <unistd.h>

static bool is_good_update_flag(uint64_t flags)
{
	return flags == BPF_ANY || flags == BPF_EXIST || flags == BPF_NOEXIST;
}

namespace bpftime
{

var_size_hash_map_impl::var_size_hash_map_impl(managed_shared_memory &memory,
					       uint32_t key_size,
					       uint32_t value_size,
					       uint32_t max_entries,
					       uint32_t flags)
	: map_impl(max_entries, bytes_vec_hasher(), std::equal_to<bytes_vec>(),
		   bi_map_allocator(memory.get_segment_manager())),
	  _key_size(key_size), _value_size(value_size),
	  _max_entries(max_entries), flags(flags),
	  key_vec(key_size, memory.get_segment_manager()),
	  value_vec(value_size, memory.get_segment_manager())
{
}

void *var_size_hash_map_impl::elem_lookup(const void *key)
{
	if (flags & BPF_F_WRONLY) {
		errno = EPERM;
		return nullptr;
	}
	SPDLOG_TRACE("Peform elem lookup of hash map");
	// Since we use lock here, we don't need to allocate key_vec and
	// value_vec
	key_vec.assign((uint8_t *)key, (uint8_t *)key + _key_size);
	if (auto itr = map_impl.find(key_vec); itr != map_impl.end()) {
		SPDLOG_TRACE("Exit elem lookup of hash map");
		return &itr->second[0];
	} else {
		SPDLOG_TRACE("Exit elem lookup of hash map");
		errno = ENOENT;
		return nullptr;
	}
}

long var_size_hash_map_impl::elem_update(const void *key, const void *value,
					 uint64_t flags)
{
	// Check flags
	if (!is_good_update_flag(flags)) {
		errno = EINVAL;
		return -1;
	}
	key_vec.assign((uint8_t *)key, (uint8_t *)key + _key_size);
	value_vec.assign((uint8_t *)value, (uint8_t *)value + _value_size);
	bool element_exists = map_impl.find(key_vec) != map_impl.end();
	// Check if the element exists...
	if (flags == BPF_NOEXIST && element_exists) {
		errno = EEXIST;
		return -1;
	}
	if (flags == BPF_EXIST && !element_exists) {
		errno = ENOENT;
		return -1;
	}
	// Ensure max_entries
	if (element_exists == false && map_impl.size() == _max_entries) {
		errno = E2BIG;
		return -1;
	}
	if (element_exists == false && (this->flags & BPF_F_RDONLY)) {
		errno = EPERM;
		return -1;
	}
	map_impl.insert_or_assign(key_vec, value_vec);
	return 0;
}

long var_size_hash_map_impl::elem_delete(const void *key)
{
	key_vec.assign((uint8_t *)key, (uint8_t *)key + _key_size);
	auto itr = map_impl.find(key_vec);
	if (itr == map_impl.end()) {
		errno = ENOENT;
		return -1;
	}
	map_impl.erase(itr);
	return 0;
}
int var_size_hash_map_impl::lookup_and_delete(const void *key, void *value_out)
{
	key_vec.assign((uint8_t *)key, (uint8_t *)key + _key_size);
	auto itr = map_impl.find(key_vec);
	if (itr == map_impl.end()) {
		errno = ENOENT;
		return -1;
	}
	memcpy(value_out, &itr->second[0], _value_size);
	map_impl.erase(itr);
	return 0;
}
int var_size_hash_map_impl::map_get_next_key(const void *key, void *next_key)
{
	if (flags & BPF_F_WRONLY) {
		errno = EPERM;
		return -1;
	}
	if (key == nullptr) {
		// nullptr means the first key
		auto itr = map_impl.begin();
		if (itr == map_impl.end()) {
			errno = ENOENT;
			return -1;
		}
		std::copy(itr->first.begin(), itr->first.end(),
			  (uint8_t *)next_key);
		return 0;
	}
	// Since we use lock here, we don't need to allocate key_vec and
	// value_vec
	key_vec.assign((uint8_t *)key, (uint8_t *)key + _key_size);

	auto itr = map_impl.find(key_vec);
	if (itr == map_impl.end()) {
		// not found, should be refer to the first key
		return map_get_next_key(nullptr, next_key);
	}
	itr++;
	if (itr == map_impl.end()) {
		// If *key* is the last element, returns -1 and *errno*
		// is set to **ENOENT**.
		errno = ENOENT;
		return -1;
	}
	std::copy(itr->first.begin(), itr->first.end(), (uint8_t *)next_key);
	return 0;
}

} // namespace bpftime
