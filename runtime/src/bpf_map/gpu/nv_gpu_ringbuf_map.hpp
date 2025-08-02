#ifndef _NV_GPU_RINGBUF_MAP_HPP
#define _NV_GPU_RINGBUF_MAP_HPP

#include "bpf_map/map_common_def.hpp"
#include "cuda.h"
#include <cstddef>
#include <functional>
#include <utility>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/unordered_map.hpp>
namespace bpftime
{
using pid_devptr_map_value_ty = std::pair<const int, CUdeviceptr>;
using pid_devptr_map_allocator = boost::interprocess::allocator<
	pid_devptr_map_value_ty,
	boost::interprocess::managed_shared_memory::segment_manager>;
using pid_devptr_map =
	boost::unordered_map<int, CUdeviceptr, int_hasher, std::equal_to<int>,
			     pid_devptr_map_allocator>;

struct ringbuf_header {
	uint64_t head;
	uint64_t tail;
};

class nv_gpu_ringbuf_map_impl {
	CUipcMemHandle gpu_mem_handle;
	/**
	    Memory layout of each thread:
	    |struct ringbuf_header|page0 (in size of value size)|page1 (in size
	   of value size)|.........
	    */

	void *server_shared_mem;
	pid_devptr_map agent_gpu_shared_mem;
	uint64_t value_size;
	uint64_t max_entries;
	uint64_t thread_count;

	bytes_vec local_buffer;

	uint64_t entry_size;

    public:
	static inline size_t calculate_total_buffer_size(size_t thread_count,
							 size_t value_size,
							 size_t max_entries)
	{
		return thread_count *
		       (value_size * max_entries + sizeof(ringbuf_header));
	}
	const static bool should_lock = true;
	nv_gpu_ringbuf_map_impl(
		boost::interprocess::managed_shared_memory &memory,

		uint64_t value_size, uint64_t max_entries,
		uint64_t thread_count);

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);
	// Only to be called at syscall server side
	int drain_data(const std::function<void(const void *)> &fn);

	CUdeviceptr get_gpu_mem_buffer()
	{
		return try_initialize_for_agent_and_get_mapped_address();
	}
	uint64_t get_max_thread_count() const
	{
		return thread_count;
	}
	virtual ~nv_gpu_ringbuf_map_impl();

	CUdeviceptr try_initialize_for_agent_and_get_mapped_address();
};
} // namespace bpftime

#endif
