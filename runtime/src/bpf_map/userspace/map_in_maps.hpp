/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _MAP_IN_MAP_HPP
#define _MAP_IN_MAP_HPP

#include <bpf_map/map_common_def.hpp>
#include "array_map.hpp"

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
	// TODO: add verify the correctness of the key
	void *elem_lookup(const void *key)
	{
		auto key_val = array_map_impl::elem_lookup(key);
		int map_id = *(int *)key_val;
		return (void *)((u_int64_t)map_id << 32);
	}
};

} // namespace bpftime

#endif // _MAP_IN_MAP_HPP
