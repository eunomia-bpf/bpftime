#ifndef BPFTIME_HASH_MAP_HPP
#define BPFTIME_HASH_MAP_HPP

#include "map_common_def.hpp"
#include <cassert>

namespace bpftime
{

using namespace boost::interprocess;

// A fixed size hash map that is used in the eBPF runtime.
// with open addressing and linear probing.
class bpftime_hash_map {
    private:
	inline size_t get_elem_offset(size_t index) const
	{
		return index * (4 + _key_size + _value_size);
	}
	size_t _num_buckets;
	size_t _key_size;
	size_t _value_size;
	// The data is stored in a contiguous memory region.
	// The layout of elem is:
	// 4 bytes is empty or not
	// key_size bytes for the key
	// value_size bytes for the value

	bytes_vec data_buffer;

    public:
	bpftime_hash_map(managed_shared_memory &memory, size_t num_buckets,
			 size_t key_size, size_t value_size)
		: _num_buckets(num_buckets), _key_size(key_size),
		  _value_size(value_size),
		  data_buffer(memory.get_segment_manager())
	{
		data_buffer.resize(num_buckets * (4 + key_size + value_size),
				   0);
	}

	size_t hash_func(const void *key)
	{
		size_t hash = 0;
		for (size_t i = 0; i < _key_size; i++) {
			hash = hash * 31 + ((uint8_t *)key)[i];
		}
		return hash;
	}

	inline bool is_empty(size_t index) const
	{
		return *(uint32_t *)&data_buffer[get_elem_offset(index)] == 0;
	}

	inline void set_empty(size_t index)
	{
		*(uint32_t *)&data_buffer[get_elem_offset(index)] = 0;
	}

	inline void set_filled(size_t index)
	{
		*(uint32_t *)&data_buffer[get_elem_offset(index)] = 1;
	}

	inline void *get_key(size_t index)
	{
		return &data_buffer[get_elem_offset(index) + 4];
	}

	inline void *get_value(size_t index)
	{
		return &data_buffer[get_elem_offset(index) + 4 + _key_size];
	}

	inline size_t get_index_of_value(const void *value)
	{
		if (value == nullptr || value < &data_buffer[0] ||
		    value >= &data_buffer[data_buffer.size()]) {
			return -1;
		}
		return (((uint8_t *)value) - &data_buffer[0]) /
		       (4 + _key_size + _value_size);
	}

	void *elem_lookup(const void *key)
	{
		size_t index = hash_func(key) % _num_buckets;
		size_t start_index = index;
		do {
			if (is_empty(index)) {
				return nullptr;
			}
			if (std::memcmp(get_key(index), key, _key_size) == 0) {
				return get_value(index);
			}
			index = (index + 1) % _num_buckets;
		} while (index != start_index);
		return nullptr;
	}

	bool elem_update(const void *key, const void *value)
	{
		size_t index = hash_func(key) % _num_buckets;
		size_t start_index = index;
		do {
			if (is_empty(index) ||
			    std::memcmp(get_key(index), key, _key_size) == 0) {
				std::memcpy(get_key(index), key, _key_size);
				std::memcpy(get_value(index), value,
					    _value_size);
				set_filled(index);
				return true;
			}
			index = (index + 1) % _num_buckets;
		} while (index != start_index);
		return false; // Hashmap is full
	}

	bool elem_delete(const void *key)
	{
		size_t index = hash_func(key) % _num_buckets;
		size_t start_index = index;
		do {
			if (is_empty(index)) {
				return false; // Key not found
			}
			if (std::memcmp(get_key(index), key, _key_size) == 0) {
				set_empty(index);
				return true;
			}
			index = (index + 1) % _num_buckets;
		} while (index != start_index);
		return false; // Key not found
	}

	size_t get_elem_count() const
	{
		size_t count = 0;
		for (size_t i = 0; i < _num_buckets; ++i) {
			if (!is_empty(i)) {
				++count;
			}
		}
		return count;
	}
};

} // namespace bpftime

#endif // BPFTIME_HASH_MAP_HPP
