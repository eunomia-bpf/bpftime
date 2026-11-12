/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _STACK_TRACE_MAP_HPP
#define _STACK_TRACE_MAP_HPP
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/unordered_map.hpp>
#include <bpf_map/map_common_def.hpp>
#include <cstdint>
#include <functional>
#include <utility>

namespace bpftime
{
namespace stack_trace_map
{

using value_ty = std::pair<const uint32_t, uint64_vec>;
using map_allocator = boost::interprocess::allocator<
	value_ty, boost::interprocess::managed_shared_memory::segment_manager>;

using stack_trace_hashmap =
	boost::unordered_map<uint32_t, uint64_vec, uint32_hasher,
			     std::equal_to<uint32_t>, map_allocator>;
template <class T> static inline uint64_t hash_stack_trace(const T &stk)
{
	using boost::hash_combine;
	size_t seed = 0;
	hash_combine(seed, stk.size());
	for (auto x : stk)
		hash_combine(seed, x);
	return seed;
}

} // namespace stack_trace_map

// implementation of array map
class stack_trace_map_impl {
	uint64_t max_stack_entries;
	uint64_t max_entries;
	stack_trace_map::stack_trace_hashmap data;

	uint64_vec key_buf;

    public:
	const static bool should_lock = true;
	stack_trace_map_impl(boost::interprocess::managed_shared_memory &memory,
			     uint32_t key_size, uint32_t value_size,
			     uint32_t max_entries);

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);

	int fill_stack_trace(const std::vector<uint64_t> &stk,
			     bool discard_old_one, bool compare_only_by_hash);
};

} // namespace bpftime
#endif
