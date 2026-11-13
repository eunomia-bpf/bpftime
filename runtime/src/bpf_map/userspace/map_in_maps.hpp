/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _MAP_IN_MAP_HPP
#define _MAP_IN_MAP_HPP

#include <bpf_map/map_common_def.hpp>
#include "array_map.hpp"
#include "var_hash_map.hpp"

namespace bpftime
{

// implementation of array map
class array_map_of_maps_impl : public array_map_impl {
    public:
	array_map_of_maps_impl(
		boost::interprocess::managed_shared_memory &memory,
		uint32_t max_entries)
		: array_map_impl(memory, sizeof(int), max_entries)
	{
	}
	void *elem_lookup(const void *key)
	{
		auto key_val = array_map_impl::elem_lookup(key);
		if (key_val == nullptr)
			return nullptr;
		int map_id = *(int *)key_val;
		return (void *)((u_int64_t)map_id);
	}
	// Syscall BPF_MAP_LOOKUP_ELEM copies value_size bytes from the returned
	// address to user space. The helper path returns the inner id as a
	// pointer-sized handle; the syscall ABI instead needs a pointer to the
	// stored 4-byte fd, so return the raw storage address. Otherwise the
	// caller's memcpy dereferences the id itself as an address (M13).
	void *elem_lookup_userspace(const void *key)
	{
		auto *stored = static_cast<int *>(array_map_impl::elem_lookup(key));
		// Preserve the ENOENT semantics of the helper path: an unset slot
		// holds fd 0 and must not be reported as a successful lookup.
		if (stored == nullptr || *stored == 0)
			return nullptr;
		return stored;
	}
};

class hash_map_of_maps_impl : public var_size_hash_map_impl {
    public:
	hash_map_of_maps_impl(
		boost::interprocess::managed_shared_memory &memory,
		uint32_t key_size, uint32_t max_entries, uint32_t flags)
		: var_size_hash_map_impl(memory, key_size, sizeof(int),
				 max_entries, flags)
	{
	}

	void *elem_lookup(const void *key)
	{
		auto value = var_size_hash_map_impl::elem_lookup(key);
		if (value == nullptr)
			return nullptr;
		int map_id = *(int *)value;
		return (void *)((u_int64_t)map_id);
	}
	// See array_map_of_maps_impl::elem_lookup_userspace: return the stored
	// 4-byte inner fd address for the syscall memcpy path instead of the
	// pointer-sized helper handle.
	void *elem_lookup_userspace(const void *key)
	{
		auto *stored = static_cast<int *>(
			var_size_hash_map_impl::elem_lookup(key));
		if (stored == nullptr || *stored == 0)
			return nullptr;
		return stored;
	}
};

} // namespace bpftime

#endif // _MAP_IN_MAP_HPP
