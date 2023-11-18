/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_KERNEL_ARRAY_MAP_HPP
#define _BPFTIME_KERNEL_ARRAY_MAP_HPP
#include <bpf_map/map_common_def.hpp>
namespace bpftime
{

// implementation of array map
class array_map_kernel_user_impl {
	bytes_vec value_data;
	uint32_t _value_size;
	uint32_t _max_entries;
	int map_fd = -1;
	int kernel_map_id = -1;

	void* mmap_ptr;

	void init_map_fd();

    public:
	const static bool should_lock = false;
	array_map_kernel_user_impl(boost::interprocess::managed_shared_memory &memory,
		       int km_id);
	~array_map_kernel_user_impl();

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);

	void *get_raw_data() const;
};

} // namespace bpftime

#endif // _BPFTIME_ARRAY_MAP_HPP
