#include "nv_gpu_shared_hash_map.hpp"
#include "bpftime_shm.hpp"
#include "bpftime_shm_internal.hpp"
#include "handler/map_handler.hpp"
#include "cuda.h"
#include "linux/bpf.h"
#include "spdlog/spdlog.h"
#include <cerrno>
#include <cstdint>
#include <stdexcept>
#include <unistd.h>

using namespace bpftime;

nv_gpu_shared_hash_map_impl::nv_gpu_shared_hash_map_impl(
	boost::interprocess::managed_shared_memory &memory,
	uint64_t max_entries, uint64_t key_size, uint64_t value_size)
	: array_map(memory, value_size, max_entries), _key_size(key_size),
	  _num_buckets(bpftime_hasher::next_prime(max_entries)), _count(0),
	  key_buffer(memory.get_segment_manager())
{
	key_buffer.resize((key_size + 4) * _num_buckets);
	pthread_spin_init(&map_lock, 0);
	SPDLOG_INFO(
		"Initializing map type of BPF_MAP_TYPE_GPU_HASH_MAP (device), key_size={}, value_size={}, max_entries={}, num_buckets={}",
		key_size, value_size, max_entries, _num_buckets);
}

void *nv_gpu_shared_hash_map_impl::key_lookup(const void *key)
{
	size_t index = bpftime_hasher::hash_func(key, _key_size) % _num_buckets;
	size_t start_index = index;

	do {
		if (is_empty(index)) {
			return nullptr;
		}
		void *current_key = get_key(index);
		if (std::memcmp(current_key, key, _key_size) == 0) {
			return current_key;
		}
		index = (index + 1) % _num_buckets;
	} while (index != start_index);
	return nullptr;
}

void *nv_gpu_shared_hash_map_impl::elem_lookup(const void *key)
{
	size_t index = bpftime_hasher::hash_func(key, _key_size) % _num_buckets;
	size_t start_index = index;
	bpftime_lock_guard guard(map_lock);

	do {
		if (is_empty(index)) {
			return nullptr;
		}
		if (std::memcmp(get_key(index), key, _key_size) == 0) {
			return array_map.elem_lookup((const void *)&index);
		}
		index = (index + 1) % _num_buckets;
	} while (index != start_index);
	return nullptr;
}

long nv_gpu_shared_hash_map_impl::elem_update(const void *key,
					      const void *value, uint64_t flags)
{
	if (!bpftime::check_update_flags(flags))
		return -1;
	size_t index = bpftime_hasher::hash_func(key, _key_size) % _num_buckets;
	size_t start_index = index;
	bpftime_lock_guard guard(map_lock);

	// Iterate over the hash map using linear probing
	do {
		if (is_empty(index)) {
			// If the current bucket is empty, insert the
			// new element
			if (_count >= _num_buckets) {
				// Reject if the hash map is full
				errno = ENOSPC;
				return -1;
			}
			// Insert the new element
			std::memcpy(get_key(index), key, _key_size);
			long ret = array_map.elem_update((const void *)&index,
							 value, flags);
			if (ret == 0) {
				set_filled(index);
				_count++; // Increase the count for the new
					  // element
				return 0;
			} else {
				return ret;
			}
		} else if (std::memcmp(get_key(index), key, _key_size) == 0) {
			// If the current bucket has a matching key,
			// update the value
			return array_map.elem_update((const void *)&index,
						     value, flags);
		}

		// Move to the next bucket
		index = (index + 1) % _num_buckets;
	} while (index != start_index);

	errno = ENOSPC;
	return -1;
}

long nv_gpu_shared_hash_map_impl::elem_delete(const void *key)
{
	size_t index = bpftime_hasher::hash_func(key, _key_size) % _num_buckets;
	size_t start_index = index;
	bpftime_lock_guard guard(map_lock);

	do {
		if (is_empty(index)) {
			errno = ENOENT;
			return -1; // Key not found
		}
		if (std::memcmp(get_key(index), key, _key_size) == 0) {
			set_empty(index);
			// Decrease count if deleting an element
			_count--;
			return 0;
		}
		index = (index + 1) % _num_buckets;
	} while (index != start_index);
	errno = ENOENT;
	return -1; // Key not found
}

int nv_gpu_shared_hash_map_impl::map_get_next_key(const void *key,
						  void *next_key)
{
	SPDLOG_TRACE("Perform map get next key of shared gpu hash map");
	if (next_key == nullptr) {
		errno = EINVAL;
		return -1;
	}
	bpftime_lock_guard guard(map_lock);

	if (key == nullptr) {
		// get the first key
		for (size_t i = 0; i < _num_buckets; i++) {
			if (is_empty(i)) {
				continue;
			}
			memcpy(next_key, get_key(i), _key_size);
			return 0;
		}
		errno = ENOENT;
		// no key found
		return -1;
	}
	// get the next key
	void *key_ptr = key_lookup(key);
	if (key_ptr == nullptr) {
		return map_get_next_key(nullptr, next_key);
	}
	// get_next_key
	size_t index = get_index_of_key(key_ptr);
	for (size_t i = index + 1; i < _num_buckets; i++) {
		if (is_empty(i)) {
			continue;
		}
		memcpy(next_key, get_key(i), _key_size);
		return 0;
	}
	// if this is the last key
	errno = ENOENT;
	return -1;
}

nv_gpu_shared_hash_map_impl::~nv_gpu_shared_hash_map_impl()
{
	pthread_spin_destroy(&map_lock);
}

CUdeviceptr
nv_gpu_shared_hash_map_impl::try_initialize_for_agent_and_get_mapped_address()
{
	return array_map.try_initialize_for_agent_and_get_mapped_address();
}
