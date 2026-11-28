/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_PER_CPU_HASH_MAP_HPP
#define _BPFTIME_PER_CPU_HASH_MAP_HPP
#include <boost/interprocess/managed_shared_memory.hpp>
#include <bpf_map/bpftime_hash_map.hpp>
#include <bpf_map/map_common_def.hpp>
#include <cstdint>

namespace bpftime
{

class per_cpu_hash_map_impl {
	bpftime_hash_map impl;
	uint32_t value_size;
	int ncpu;
	// Per-CPU values are 8-byte aligned, the layout the bpf syscall uses.
	uint32_t stride() const { return (value_size + 7U) & ~7U; }

    public:
	const static bool should_lock = false;

	per_cpu_hash_map_impl(boost::interprocess::managed_shared_memory &memory,
			      uint32_t key_size, uint32_t value_size,
			      uint32_t max_entries);
	per_cpu_hash_map_impl(boost::interprocess::managed_shared_memory &memory,
			      uint32_t key_size, uint32_t value_size,
			      uint32_t max_entries, int ncpu);
	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);

	void *elem_lookup_userspace(const void *key);

	long elem_update_userspace(const void *key, const void *value,
				   uint64_t flags);

	long elem_delete_userspace(const void *key);
	long lookup_and_delete_userspace(const void *key, void *value,
					 bool from_syscall = true);
	uint32_t get_value_size() const
	{
		return value_size;
	}
	int getncpu() const
	{
		return ncpu;
	}
};
} // namespace bpftime

#endif
