#ifndef _NV_GPU_SHARED_ARRAY_HOST_MAP_HPP
#define _NV_GPU_SHARED_ARRAY_HOST_MAP_HPP

#include "bpf_map/map_common_def.hpp"
#include "cuda.h"
#include "handler/handler_manager.hpp"
#include <cstdint>

namespace bpftime
{

using pid_devptr_map_value_ty = std::pair<const int, CUdeviceptr>;
using pid_devptr_map_allocator =
	allocator<pid_devptr_map_value_ty,
		  managed_shared_memory::segment_manager>;
using pid_devptr_map =
	boost::unordered_map<int, CUdeviceptr, int_hasher, std::equal_to<int>,
			     pid_devptr_map_allocator>;

// GPU shared array map using boost::interprocess + cudaHostRegister
// This is for platforms that don't support CUDA IPC (e.g., Tegra)
class nv_gpu_shared_array_host_map_impl {
	// Data buffer in boost::interprocess shared memory
	// Layout: char BUF[MAX_ENTRIES][VALUE_SIZE]
	bytes_vec data_buffer;

	// Cache for agent-side GPU device pointers (after cudaHostRegister)
	pid_devptr_map agent_gpu_shared_mem;

	uint64_t value_size;
	uint64_t max_entries;

    public:
	const static bool should_lock = true;

	nv_gpu_shared_array_host_map_impl(
		boost::interprocess::managed_shared_memory &memory,
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
	virtual ~nv_gpu_shared_array_host_map_impl();

	CUdeviceptr try_initialize_for_agent_and_get_mapped_address();
};
} // namespace bpftime

#endif
