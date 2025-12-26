/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#ifndef _BPFTIME_KERNEL_GPU_MAP_HPP
#define _BPFTIME_KERNEL_GPU_MAP_HPP
#include <bpf_map/map_common_def.hpp>
#include <boost/unordered_map.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include "cuda.h"

namespace bpftime
{
using pid_devptr_map_value_ty = std::pair<const int, CUdeviceptr>;
using pid_devptr_map_allocator = boost::interprocess::allocator<
	pid_devptr_map_value_ty,
	boost::interprocess::managed_shared_memory::segment_manager>;
using pid_devptr_map =
	boost::unordered_map<int, CUdeviceptr, bpftime::int_hasher,
			     std::equal_to<int>, pid_devptr_map_allocator>;

// implementation of array map
class array_map_kernel_gpu_impl {
	bytes_vec value_data;
	uint32_t _value_size;
	uint32_t _max_entries;
	int map_fd = -1;
	int kernel_map_id = -1;

	void *mmap_ptr;

	void init_map_fd();

    public:
	const static bool should_lock = false;
	array_map_kernel_gpu_impl(
		boost::interprocess::managed_shared_memory &memory, int km_id,
		uint32_t value_size, uint32_t max_entries);
	~array_map_kernel_gpu_impl();

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);

	void *get_raw_data() const;

	CUdeviceptr try_initialize_for_agent_and_get_mapped_address();

    private:
	pid_devptr_map agent_gpu_shared_mem;
};

} // namespace bpftime

#endif // _BPFTIME_ARRAY_MAP_HPP
