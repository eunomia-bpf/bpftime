/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_PER_CPU_HASH_MAP_HPP
#define _BPFTIME_PER_CPU_HASH_MAP_HPP
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/smart_ptr/unique_ptr.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <bpf_map/map_common_def.hpp>
#include <boost/unordered/unordered_map.hpp>
#include <cstdint>

namespace bpftime
{

class per_cpu_hash_map_impl {
	using bi_map_value_ty = std::pair<const bytes_vec, bytes_vec>;
	using bi_map_allocator = boost::interprocess::allocator<
		bi_map_value_ty,
		boost::interprocess::managed_shared_memory::segment_manager>;
	using shm_hash_map =
		boost::unordered_map<bytes_vec, bytes_vec, bytes_vec_hasher,
				     std::equal_to<bytes_vec>, bi_map_allocator>;

	using shm_hash_map_vec_allocator = boost::interprocess::allocator<
		shm_hash_map,
		boost::interprocess::managed_shared_memory::segment_manager>;
	using shm_hash_map_vec =
		boost::interprocess::vector<shm_hash_map,
					    shm_hash_map_vec_allocator>;

	using bytes_vec_vec_allocator = boost::interprocess::allocator<
		bytes_vec,
		boost::interprocess::managed_shared_memory::segment_manager>;
	using bytes_vec_vec =
		boost::interprocess::vector<bytes_vec, bytes_vec_vec_allocator>;

	shm_hash_map impl;
	uint32_t key_size;
	uint32_t value_size;
	int ncpu;
	uint32_t _max_entries;

	bytes_vec value_template;
	bytes_vec_vec key_templates, single_value_templates;

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
	long lookup_and_delete_userspace(const void *key, void *value);
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
