#ifndef _NV_GPU_SHARED_HASH_MAP_HPP
#define _NV_GPU_SHARED_HASH_MAP_HPP

#include "bpf_map/map_common_def.hpp"
#include "bpf_map/bpftime_hash_map.hpp"
#include "bpf_map/gpu/nv_gpu_shared_array_map.hpp"
#include "cuda.h"
#include "handler/handler_manager.hpp"
#include <cstdint>

namespace bpftime
{

// Similar to the `bpftime_hash_map` with open addressing and linear probing,
// For simplicity, we store the key and value separately.
class nv_gpu_shared_hash_map_impl {
	// the underlying array map to store the value
	nv_gpu_shared_array_map_impl array_map;
	uint64_t _key_size;
	uint64_t _num_buckets;
	size_t _count; // Current number of elements

	// Host-side buffer for all of keys
	bytes_vec key_buffer;

	// The `array_map` will be locked in bpf_map_handler, so we ensure the
	// thread-safety internally.
	mutable pthread_spinlock_t map_lock;

	inline size_t get_elem_offset(size_t index) const
	{
		return index * (4 + _key_size);
	}

	inline bool is_empty(size_t index) const
	{
		return *(uint32_t *)(uintptr_t)&key_buffer[get_elem_offset(
			       index)] == 0;
	}

	inline void set_empty(size_t index)
	{
		*(uint32_t *)(uintptr_t)&key_buffer[get_elem_offset(index)] = 0;
	}

	inline void set_filled(size_t index)
	{
		*(uint32_t *)(uintptr_t)&key_buffer[get_elem_offset(index)] = 1;
	}

	inline void *get_key(size_t index)
	{
		return &key_buffer[get_elem_offset(index) + 4];
	}

	inline size_t get_index_of_key(const void *key)
	{
		if (key == nullptr || key < &key_buffer[0] ||
		    key >= key_buffer.data() + key_buffer.size()) {
			return -1;
		}
		return (((uint8_t *)key) - &key_buffer[0]) / (4 + _key_size);
	}

	void *key_lookup(const void *key);

    public:
	// The `array_map` will be locked in bpf_map_handler, so we ensure the
	// thread-safety internally.
	const static bool should_lock = false;

	nv_gpu_shared_hash_map_impl(
		boost::interprocess::managed_shared_memory &memory,
		uint64_t max_entries, uint64_t key_size, uint64_t value_size);

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
	virtual ~nv_gpu_shared_hash_map_impl();

	CUdeviceptr try_initialize_for_agent_and_get_mapped_address();
};
} // namespace bpftime

#endif
