/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_KERNEL_HASH_MAP_HPP
#define _BPFTIME_KERNEL_HASH_MAP_HPP
#include <boost/container_hash/hash_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <bpf_map/map_common_def.hpp>
#include <boost/unordered/unordered_map.hpp>
#include <boost/functional/hash.hpp>
namespace bpftime
{

using namespace boost::interprocess;

// implementation of hash map
class hash_map_kernel_user_impl {

	uint32_t _key_size;
	uint32_t _value_size;

	bytes_vec key_vec;
	bytes_vec value_vec;

	int kernel_map_id = -1;
	int map_fd = -1;

	void init_map_fd();

    public:
	const static bool should_lock = false;
	hash_map_kernel_user_impl(managed_shared_memory &memory, int km_id);
	~hash_map_kernel_user_impl();

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);
};
} // namespace bpftime

#endif // _BPFTIME_KERNEL_HASH_MAP_HPP
