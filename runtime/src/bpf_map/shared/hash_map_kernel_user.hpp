/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_KERNEL_HASH_MAP_HPP
#define _BPFTIME_KERNEL_HASH_MAP_HPP
#include "bpftime_shm_safe.hpp"
#include <bpf_map/map_common_def.hpp>

namespace bpftime
{

class hash_map_kernel_user_impl {
	uint32_t _key_size = 0;
	uint32_t _value_size = 0;
	bytes_vec key_vec;
	bytes_vec value_vec;
	int kernel_map_id = -1;
	shm_local_key local_key_{};

    public:
	const static bool should_lock = false;
	hash_map_kernel_user_impl(const shm_init_ctx &ctx, int km_id);
	~hash_map_kernel_user_impl();

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);
};

} // namespace bpftime

#endif // _BPFTIME_KERNEL_HASH_MAP_HPP
