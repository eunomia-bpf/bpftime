/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_PER_CPU_ARRAY_MAP_HPP
#define _BPFTIME_PER_CPU_ARRAY_MAP_HPP
#include "bpf_map/map_common_def.hpp"
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/smart_ptr/unique_ptr.hpp>
#include <boost/interprocess/containers/vector.hpp>

#include <cstdint>
namespace bpftime
{

class per_cpu_array_map_impl {
	bytes_vec data;

	int ncpu;
	uint32_t value_size;
	uint32_t max_ent;
	uint8_t *data_at(size_t idx, size_t cpu)
	{
		return data.data() + idx * value_size * ncpu + cpu * value_size;
	}

    public:
	const static bool should_lock = false;

	per_cpu_array_map_impl(
		boost::interprocess::managed_shared_memory &memory,
		uint32_t value_size, uint32_t max_entries, uint32_t ncpu);
	per_cpu_array_map_impl(
		boost::interprocess::managed_shared_memory &memory,
		uint32_t value_size, uint32_t max_entries);

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);

	void *elem_lookup_userspace(const void *key);

	long elem_update_userspace(const void *key, const void *value,
				   uint64_t flags);

	long elem_delete_userspace(const void *key);
};
} // namespace bpftime

#endif
