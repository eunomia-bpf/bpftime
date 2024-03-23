/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _PROG_ARRAY_HPP
#define _PROG_ARRAY_HPP
#include <cstdint>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>

namespace bpftime
{

using i32_vec_allocator = boost::interprocess::allocator<
	int32_t, boost::interprocess::managed_shared_memory::segment_manager>;
using i32_vec = boost::interprocess::vector<int32_t, i32_vec_allocator>;
class prog_array_map_impl {
    private:
	// accept fd, but store id
	// maps might be shared among processes, fd are process independent, but
	// ids are shared
	i32_vec data;

    public:
	const static bool should_lock = false;
	prog_array_map_impl(boost::interprocess::managed_shared_memory &memory,
			    uint32_t key_size, uint32_t value_size,
			    uint32_t max_entries);

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);
};
} // namespace bpftime
#endif
