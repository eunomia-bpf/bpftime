#ifndef _NV_GPU_SHARED_ARRAY_MAP_HPP
#define _NV_GPU_SHARED_ARRAY_MAP_HPP

#include "bpf_map/map_common_def.hpp"
#include "bpftime_shm_safe.hpp"
#include "cuda.h"
#include "handler/handler_manager.hpp"
#include <cstdint>

namespace bpftime
{

class nv_gpu_shared_array_map_impl {
	// Single-copy device buffer: char BUF[MAX_ENTRIES][VALUE_SIZE]
	CUipcMemHandle gpu_mem_handle;
	CUdeviceptr server_gpu_shared_mem;
	uint64_t value_size;
	uint64_t max_entries;
	shm_local_key local_key_{};

	// Host-side staging buffer for single entry
	bytes_vec value_buffer;

    public:
	const static bool should_lock = true;

	nv_gpu_shared_array_map_impl(const shm_init_ctx &ctx,
				     uint64_t value_size, uint64_t max_entries);

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);

	CUdeviceptr get_gpu_mem_buffer()
	{
		return try_initialize_for_agent_and_get_mapped_address();
	}
	uint64_t get_max_thread_count() const
	{
		return 1;
	}
	~nv_gpu_shared_array_map_impl();

	CUdeviceptr try_initialize_for_agent_and_get_mapped_address();
};
} // namespace bpftime

#endif
