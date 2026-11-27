/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _MAP_IN_MAP_HPP
#define _MAP_IN_MAP_HPP

#include <bpf_map/map_common_def.hpp>
#include <cerrno>
#include <cstdint>
#include "array_map.hpp"
#include "fix_hash_map.hpp"

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
	// caller's memcpy would dereference the id itself as an invalid address.
	void *elem_lookup_userspace(const void *key)
	{
		auto *stored = static_cast<int *>(array_map_impl::elem_lookup(key));
		// ENOENT on a null/zero slot, matching kernel map-in-map semantics:
		// a never-updated slot holds 0 and BPF_MAP_LOOKUP_ELEM returns
		// ENOENT. Inner map fds come from BPF_MAP_CREATE via open_fake_fd,
		// which never yields 0/1/2, so 0 unambiguously means "no inner map".
		if (stored == nullptr || *stored == 0)
			return nullptr;
		return stored;
	}
};

class hash_map_of_maps_impl : public fix_size_hash_map_impl {
    public:
	hash_map_of_maps_impl(
		boost::interprocess::managed_shared_memory &memory,
		uint32_t key_size, uint32_t max_entries, uint32_t /*flags*/)
		: fix_size_hash_map_impl(memory, max_entries, key_size,
					 sizeof(int))
	{
	}

	void *elem_lookup(const void *key)
	{
		int map_id = 0;
		if (!fix_size_hash_map_impl::elem_lookup_copy(key, &map_id)) {
			errno = ENOENT;
			return nullptr;
		}
		return (void *)((u_int64_t)map_id);
	}
	// See array_map_of_maps_impl::elem_lookup_userspace: return a stable
	// non-zero 4-byte snapshot address so the syscall caller cannot observe
	// a concurrent delete or zero write between validation and memcpy.
	void *elem_lookup_userspace(const void *key)
	{
		int map_id = 0;
		if (!fix_size_hash_map_impl::elem_lookup_copy(key, &map_id) ||
		    map_id == 0) {
			errno = ENOENT;
			return nullptr;
		}
		static thread_local int snapshot = 0;
		snapshot = map_id;
		return &snapshot;
	}
};

} // namespace bpftime

#endif // _MAP_IN_MAP_HPP
