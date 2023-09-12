#include "bpftime_handler.hpp"

#include <algorithm>
#include <array>
#include <boost/container/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/smart_ptr/unique_ptr.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <iostream>
#include <string>
#include <utility>

namespace bpftime
{

using bytes_vec_allocator =
	allocator<uint8_t, managed_shared_memory::segment_manager>;
using bytes_vec = boost::container::vector<uint8_t, bytes_vec_allocator>;
using namespace boost::interprocess;

struct BytesVecCompFunctor {
	bool operator()(const bytes_vec &a, const bytes_vec &b) const
	{
		if (a.size() != b.size())
			return a.size() < b.size();
		for (size_t i = 0; i < a.size(); i++) {
			if (a[i] == b[i])
				continue;
			return a[i] < b[i];
		}
		return false;
	}
};

// implementation of hash map
class hash_map_impl {
	using bi_map_value_ty = std::pair<const bytes_vec, bytes_vec>;
	using bi_map_allocator =
		allocator<bi_map_value_ty,
			  managed_shared_memory::segment_manager>;
	boost::interprocess::map<bytes_vec, bytes_vec, BytesVecCompFunctor,
				 bi_map_allocator>
		map_impl;
	uint32_t _key_size;
	uint32_t _value_size;

	bytes_vec key_vec;
	bytes_vec value_vec;

    public:
	hash_map_impl(managed_shared_memory &memory, uint32_t key_size,
		      uint32_t value_size)
		: map_impl(BytesVecCompFunctor(),
			   bi_map_allocator(memory.get_segment_manager())),
		  _key_size(key_size), _value_size(value_size),
		  key_vec(key_size, memory.get_segment_manager()),
		  value_vec(value_size, memory.get_segment_manager())
	{
	}

	void *elem_lookup(const void *key)
	{
		key_vec.assign((uint8_t *)key, (uint8_t *)key + _key_size);

		if (auto itr = map_impl.find(key_vec); itr != map_impl.end()) {
			return &itr->second[0];
		} else {
			errno = ENOENT;
			return nullptr;
		}
	}

	long elem_update(const void *key, const void *value, uint64_t flags)
	{
		key_vec.assign((uint8_t *)key, (uint8_t *)key + _key_size);
		value_vec.assign((uint8_t *)value,
				 (uint8_t *)value + _value_size);
		map_impl.emplace(key_vec, value_vec);
		return 0;
	}

	long elem_delete(const void *key)
	{
		key_vec.assign((uint8_t *)key, (uint8_t *)key + _key_size);
		map_impl.erase(key_vec);
		return 0;
	}

	int bpf_map_get_next_key(const void *key, void *next_key)
	{
		if (key == nullptr) {
			// nullptr means the first key
			auto itr = map_impl.begin();
			if (itr == map_impl.end()) {
				errno = ENOENT;
				return -1;
			}
			std::copy(itr->first.begin(), itr->first.end(),
				  (uint8_t *)next_key);
			return 0;
		}
		key_vec.assign((uint8_t *)key, (uint8_t *)key + _key_size);
		auto itr = map_impl.find(key_vec);
		if (itr == map_impl.end()) {
			// not found, should be refer to the first key
			return bpf_map_get_next_key(nullptr, next_key);
		}
		itr++;
		if (itr == map_impl.end()) {
			// If *key* is the last element, returns -1 and *errno*
			// is set to **ENOENT**.
			errno = ENOENT;
			return -1;
		}
		std::copy(itr->first.begin(), itr->first.end(),
			  (uint8_t *)next_key);
		return 0;
	}
};

// implementation of array map
class array_map_impl {
	bytes_vec data;
	uint32_t _value_size;
	uint32_t _max_entries;

    public:
	array_map_impl(managed_shared_memory &memory, uint32_t value_size,
		       uint32_t max_entries)
		: data(value_size * max_entries, memory.get_segment_manager())
	{
		this->_value_size = value_size;
		this->_max_entries = max_entries;
	}

	void *elem_lookup(const void *key)
	{
		auto key_val = *(uint32_t *)key;
		if (key_val >= _max_entries) {
			errno = ENOENT;
			return nullptr;
		}
		return &data[key_val * _value_size];
	}

	long elem_update(const void *key, const void *value, uint64_t flags)
	{
		auto key_val = *(uint32_t *)key;
		if (key_val >= _max_entries) {
			errno = ENOENT;
			return -1;
		}
		std::copy((uint8_t *)value, (uint8_t *)value + _value_size,
			  &data[key_val * _value_size]);
		return 0;
	}

	long elem_delete(const void *key)
	{
		auto key_val = *(uint32_t *)key;
		if (key_val >= _max_entries) {
			errno = ENOENT;
			return -1;
		}
		std::fill(&data[key_val * _value_size],
			  &data[key_val * _value_size] + _value_size, 0);
		return 0;
	}

	int bpf_map_get_next_key(const void *key, void *next_key)
	{
		if (_max_entries == 0 || *(uint32_t *)key == _max_entries - 1) {
			errno = ENOENT;
			return -1;
		}
		if (key == nullptr || *(uint32_t *)key >= _max_entries - 1) {
			*(uint32_t *)next_key = 0;
			return 0;
		}
		auto key_val = *(uint32_t *)key;
		*(uint32_t *)next_key = key_val + 1;
		return 0;
	}
};

const void *bpf_map_handler::map_lookup_elem(const void *key) const
{
	switch (type) {
	case BPF_MAP_TYPE_HASH:
		return static_cast<hash_map_impl *>(map_impl_ptr.get())
			->elem_lookup(key);
	case BPF_MAP_TYPE_ARRAY:
		return static_cast<array_map_impl *>(map_impl_ptr.get())
			->elem_lookup(key);
	default:
		assert(false || "Unsupported map type");
	}
	return 0;
}

long bpf_map_handler::map_update_elem(const void *key, const void *value,
				      uint64_t flags) const
{
	switch (type) {
	case BPF_MAP_TYPE_HASH:
		return static_cast<hash_map_impl *>(map_impl_ptr.get())
			->elem_update(key, value, flags);
	case BPF_MAP_TYPE_ARRAY:
		return static_cast<array_map_impl *>(map_impl_ptr.get())
			->elem_update(key, value, flags);
	default:
		assert(false || "Unsupported map type");
	}
	return 0;
}

int bpf_map_handler::bpf_map_get_next_key(const void *key, void *next_key) const
{
	switch (type) {
	case BPF_MAP_TYPE_HASH:
		return static_cast<hash_map_impl *>(map_impl_ptr.get())
			->bpf_map_get_next_key(key, next_key);
	case BPF_MAP_TYPE_ARRAY:
		return static_cast<array_map_impl *>(map_impl_ptr.get())
			->bpf_map_get_next_key(key, next_key);
	default:
		assert(false || "Unsupported map type");
	}
	return 0;
}

long bpf_map_handler::map_delete_elem(const void *key) const
{
	switch (type) {
	case BPF_MAP_TYPE_HASH:
		return static_cast<hash_map_impl *>(map_impl_ptr.get())
			->elem_delete(key);
	case BPF_MAP_TYPE_ARRAY:
		return static_cast<array_map_impl *>(map_impl_ptr.get())
			->elem_delete(key);
	default:
		assert(false || "Unsupported map type");
	}
	return 0;
}

int bpf_map_handler::map_init(managed_shared_memory &memory)
{
	auto container_name = get_container_name();
	switch (type) {
	case BPF_MAP_TYPE_HASH:
		map_impl_ptr = memory.construct<hash_map_impl>(
			container_name.c_str())(memory, key_size, value_size);
		return 0;
	case BPF_MAP_TYPE_ARRAY:
		map_impl_ptr = memory.construct<array_map_impl>(
			container_name.c_str())(memory, value_size,
						max_entries);
		return 0;
	default:
		assert(false || "Unsupported map type");
	}
	return 0;
}

void bpf_map_handler::map_free(managed_shared_memory &memory)
{
	auto container_name = get_container_name();
	switch (type) {
	case BPF_MAP_TYPE_HASH:
		memory.destroy<hash_map_impl>(container_name.c_str());
		break;
	case BPF_MAP_TYPE_ARRAY:
		memory.destroy<array_map_impl>(container_name.c_str());
		break;
	default:
		assert(false || "Unsupported map type");
	}
	map_impl_ptr = nullptr;
	return;
}

} // namespace bpftime
