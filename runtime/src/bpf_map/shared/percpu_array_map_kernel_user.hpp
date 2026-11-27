/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_PERCPU_ARRAY_MAP_KERNEL_USER_HPP
#define _BPFTIME_PERCPU_ARRAY_MAP_KERNEL_USER_HPP
#include "bpftime_shm_safe.hpp"
#include <bpf_map/map_common_def.hpp>
#include <cstdint>

namespace bpftime
{

class percpu_array_map_kernel_user_impl {
	uint32_t _value_size = 0;
	uint32_t _max_entries = 0;
	int kernel_map_id = -1;
	int ncpu = 1;
	shm_local_key local_key_{};
	bytes_vec value_data;

    public:
	const static bool should_lock = false;

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);
	percpu_array_map_kernel_user_impl(const shm_init_ctx &ctx,
					  int kernel_map_id);
	~percpu_array_map_kernel_user_impl();
};
} // namespace bpftime

#endif
