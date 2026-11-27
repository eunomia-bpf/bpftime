/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _MAP_IN_MAP_HPP
#define _MAP_IN_MAP_HPP

#include <bpf_map/map_common_def.hpp>
#include "array_map.hpp"
#include <bpf_map/bpftime_hash_map.hpp>
#include <cerrno>

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

// Values are 4-byte inner map ids, so HASH_OF_MAPS uses the preallocated
// bpftime_hash_map like HASH: lock-free lookups and recover-and-continue write
// stripes, with no Boost node allocation and no map_lock.
class hash_map_of_maps_impl {
	bpftime_hash_map map_impl;

    public:
	static constexpr bool should_lock = false;
	hash_map_of_maps_impl(
		boost::interprocess::managed_shared_memory &memory,
		uint32_t key_size, uint32_t max_entries, uint32_t /*flags*/)
		: map_impl(memory, max_entries, key_size, sizeof(int))
	{
	}

	void *elem_lookup(const void *key)
	{
		int map_id = 0;
		if (!map_impl.elem_lookup_copy(key, &map_id))
			return errno = ENOENT, nullptr;
		return (void *)((u_int64_t)map_id);
	}
	// Slots are recycled after a delete, so copy the id while the slot
	// generation is still checked instead of returning the slot address.
	void *elem_lookup_userspace(const void *key)
	{
		static thread_local int snapshot;
		if (!map_impl.elem_lookup_copy(key, &snapshot) || snapshot == 0)
			return nullptr;
		return &snapshot;
	}
	// Only the syscall path writes: the handler rejects helper updates and
	// deletes of map-in-map types.
	long elem_update(const void *key, const void *value, uint64_t flags)
	{
		if (flags > 2 /* BPF_EXIST */)
			return errno = EINVAL, -1;
		return map_impl.upsert(key, value, flags, true, 0, sizeof(int));
	}
	long elem_delete(const void *key)
	{
		return map_impl.erase(key, nullptr, true);
	}
	int map_get_next_key(const void *key, void *next_key)
	{
		return map_impl.map_get_next_key(key, next_key);
	}
};

} // namespace bpftime

#endif // _MAP_IN_MAP_HPP
