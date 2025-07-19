#ifndef _NV_GPU_ARRAY_MAP_HPP
#define _NV_GPU_ARRAY_MAP_HPP

#include "bpf_map/map_common_def.hpp"
#include <cstdint>
namespace bpftime
{
class nv_gpu_array_map_impl {
	// char BUF[MAX_ENTRIES][THREAD_COUNT][VALUE_SIZE]
	void *gpu_mem_buffer;
	uint64_t value_size;
	uint64_t max_entries;
	uint64_t thread_count;

	uint64_t entry_size;

	bytes_vec value_buffer;

    public:
	const static bool should_lock = false;
	nv_gpu_array_map_impl(
		boost::interprocess::managed_shared_memory &memory,
		void *gpu_mem_buffer, uint32_t value_size, uint32_t max_entries,
		uint32_t thread_count);

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);

	void *get_gpu_mem_buffer() const
	{
		return gpu_mem_buffer;
	}
	virtual ~nv_gpu_array_map_impl();
};
} // namespace bpftime

#endif
