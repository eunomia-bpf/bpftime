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
#include <cstring>
#include "linux/bpf.h"
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
	uint32_t flags;

    public:
	hash_map_of_maps_impl(
		boost::interprocess::managed_shared_memory &memory,
		uint32_t key_size, uint32_t max_entries, uint32_t flags)
		: fix_size_hash_map_impl(memory, max_entries, key_size,
					 sizeof(int)),
		  flags(flags)
	{
	}

	void *elem_lookup(const void *key)
	{
		uint64_t raw_val = 0;
		if (!fix_size_hash_map_impl::elem_lookup_copy(key, &raw_val)) {
			errno = ENOENT;
			return nullptr;
		}
		int map_id = 0;
		std::memcpy(&map_id, &raw_val, sizeof(map_id));
		return (void *)((u_int64_t)map_id);
	}
	// See array_map_of_maps_impl::elem_lookup_userspace: return a stable
	// non-zero 4-byte snapshot address so the syscall caller cannot observe
	// a concurrent delete or zero write between validation and memcpy.
	void *elem_lookup_userspace(const void *key)
	{
		if (flags & BPF_F_WRONLY) {
			errno = EPERM;
			return nullptr;
		}
		uint64_t raw_val = 0;
		if (!fix_size_hash_map_impl::elem_lookup_copy(key, &raw_val)) {
			errno = ENOENT;
			return nullptr;
		}
		int map_id = 0;
		std::memcpy(&map_id, &raw_val, sizeof(map_id));
		if (map_id == 0) {
			errno = ENOENT;
			return nullptr;
		}
		static thread_local int snapshot = 0;
		snapshot = map_id;
		return &snapshot;
	}

	long elem_update(const void *key, const void *value,
			 uint64_t update_flags, bool from_syscall = true)
	{
		if (from_syscall && (flags & BPF_F_RDONLY)) {
			errno = EPERM;
			return -1;
		}
		return fix_size_hash_map_impl::elem_update(key, value,
							   update_flags,
							   from_syscall);
	}

	long elem_delete(const void *key, bool from_syscall = true)
	{
		if (from_syscall && (flags & BPF_F_RDONLY)) {
			errno = EPERM;
			return -1;
		}
		return fix_size_hash_map_impl::elem_delete(key, from_syscall);
	}

	int lookup_and_delete(const void *key, void *value_out,
			      bool from_syscall = false)
	{
		if (from_syscall && (flags & (BPF_F_RDONLY | BPF_F_WRONLY))) {
			errno = EPERM;
			return -1;
		}
		return fix_size_hash_map_impl::lookup_and_delete(
			key, value_out, from_syscall);
	}

	int map_get_next_key(const void *key, void *next_key,
			     bool from_syscall = false)
	{
		if (from_syscall && (flags & BPF_F_WRONLY)) {
			errno = EPERM;
			return -1;
		}
		return fix_size_hash_map_impl::map_get_next_key(key, next_key);
	}
};

} // namespace bpftime

#endif // _MAP_IN_MAP_HPP
