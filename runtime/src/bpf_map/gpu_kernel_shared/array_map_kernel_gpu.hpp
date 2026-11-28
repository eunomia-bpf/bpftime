/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_KERNEL_GPU_MAP_HPP
#define _BPFTIME_KERNEL_GPU_MAP_HPP
#include "bpftime_shm_safe.hpp"
#include "cuda.h"
#include <bpf_map/map_common_def.hpp>

namespace bpftime
{

// implementation of kernel-GPU shared array map
class array_map_kernel_gpu_impl {
	bytes_vec value_data;
	uint32_t _value_size;
	uint32_t _max_entries;
	int kernel_map_id = -1;
	bool has_kernel_map = false;
	shm_local_key local_key_{};

    public:
	const static bool should_lock = false;
	array_map_kernel_gpu_impl(const shm_init_ctx &ctx, int km_id,
				  uint32_t value_size, uint32_t max_entries);
	~array_map_kernel_gpu_impl();

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);

	void *get_raw_data() const;

	CUdeviceptr try_initialize_for_agent_and_get_mapped_address();
};

} // namespace bpftime

#endif // _BPFTIME_KERNEL_GPU_MAP_HPP
