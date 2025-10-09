#ifndef _NV_GPU_SHARED_ARRAY_MAP_HPP
#define _NV_GPU_SHARED_ARRAY_MAP_HPP

#include "bpf_map/map_common_def.hpp"
#include <cstdint>
#include <boost/interprocess/managed_shared_memory.hpp>

namespace bpftime
{

class nv_gpu_shared_array_map_impl {
	uint64_t value_size;
	uint64_t max_entries;

	// Host-side backing storage inside bpftime shared memory
	bytes_vec shared_area;
	// Host-side staging buffer for single entry
	bytes_vec value_buffer;

	static inline uint64_t align_up(uint64_t x, uint64_t align)
	{
		return (x + align - 1) / align * align;
	}
	inline uint8_t *shared_base_ptr()
	{
		return shared_area.empty() ? nullptr : shared_area.data();
	}
	inline const uint8_t *shared_base_ptr() const
	{
		return shared_area.empty() ? nullptr : shared_area.data();
	}
	inline uint64_t locks_bytes() const
	{
		return max_entries * sizeof(int);
	}
	inline uint64_t dirty_bytes() const
	{
		return max_entries * sizeof(int);
	}
	inline uint64_t values_offset() const
	{
		return align_up(locks_bytes() + dirty_bytes(), 8);
	}
	inline uint8_t *values_region_base()
	{
		return shared_base_ptr() + values_offset();
	}
	inline const uint8_t *values_region_base() const
	{
		return shared_base_ptr() + values_offset();
	}

    public:
	const static bool should_lock = true;

	nv_gpu_shared_array_map_impl(
		boost::interprocess::managed_shared_memory &memory,
		uint64_t value_size, uint64_t max_entries);

	void *elem_lookup(const void *key);

	long elem_update(const void *key, const void *value, uint64_t flags);

	long elem_delete(const void *key);

	int map_get_next_key(const void *key, void *next_key);

	void *get_gpu_mem_buffer()
	{
		// device direct access is disabled for GPU mode; host path only
		return nullptr;
	}
	uint64_t get_max_thread_count() const
	{
		return 1;
	}
	virtual ~nv_gpu_shared_array_map_impl();
};
} // namespace bpftime

#endif
