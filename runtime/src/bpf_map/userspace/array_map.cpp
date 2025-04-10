/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2022, eunomia-bpf org
 * All rights reserved.
 */
#include "bpf_map/map_common_def.hpp"
#include "linux/bpf.h"
#include <bpf_map/userspace/array_map.hpp>
#include <cerrno>

namespace bpftime
{

void *array_map_impl::get_raw_data() const
{
	return (void *)data.data();
}
array_map_impl::array_map_impl(
	boost::interprocess::managed_shared_memory &memory, uint32_t value_size,
	uint32_t max_entries)
	: data(value_size * max_entries, memory.get_segment_manager())
{
	this->_value_size = value_size;
	this->_max_entries = max_entries;
}
void *array_map_impl::elem_lookup(const void *key)
{
	auto key_val = *(uint32_t *)key;
	if (key_val >= _max_entries) {
		errno = ENOENT;
		return nullptr;
	}
	return &data[key_val * _value_size];
}

long array_map_impl::elem_update(const void *key, const void *value,
				 uint64_t flags)
{
	if (!check_update_flags(flags))
		return -1;
	auto key_val = *(uint32_t *)key;
	if (key_val < _max_entries && flags == BPF_NOEXIST) {
		errno = EEXIST;
		return -1;
	}

	if (key_val >= _max_entries) {
		errno = E2BIG;
		return -1;
	}
	std::copy((uint8_t *)value, (uint8_t *)value + _value_size,
		  &data[key_val * _value_size]);
	return 0;
}

long array_map_impl::elem_delete(const void *key)
{
	auto key_val = *(uint32_t *)key;
	// kernel tests says element in an array map can't be deleted...
	errno = EINVAL;
	return -1;
	// if (key_val >= _max_entries) {
	// 	errno = ENOENT;
	// 	return -1;
	// }
	// std::fill(&data[key_val * _value_size],
	// 	  &data[key_val * _value_size] + _value_size, 0);
	// return 0;
}

int array_map_impl::map_get_next_key(const void *key, void *next_key)
{
	// Not found
	if (key == nullptr || *(uint32_t *)key >= _max_entries) {
		*(uint32_t *)next_key = 0;
		return 0;
	}
	uint32_t deref_key = *(uint32_t *)key;
	// Last element
	if (deref_key == _max_entries - 1) {
		errno = ENOENT;
		return -1;
	}
	auto key_val = *(uint32_t *)key;
	*(uint32_t *)next_key = key_val + 1;
	return 0;
}
} // namespace bpftime
